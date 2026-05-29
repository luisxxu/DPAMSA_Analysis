"""
ACER — Actor-Critic with Experience Replay
==========================================
Implements the key algorithmic components of Wang et al. (2016)
"Sample Efficient Actor-Critic with Experience Replay":

  1. Retrace(λ) off-policy Q-value targets
     Backward recursion:
       Q^ret_{T-1} = r_{T-1}                          (terminal bootstrap = 0)
       Q^ret_t     = r_t + γ·done_t·
                     (V(s_{t+1}) + c_{t+1}·(Q^ret_{t+1} − Q(s_{t+1},a_{t+1})))
     where c_t = min(c_bar, π(a_t|s_t) / μ(a_t|s_t)).

  2. Off-policy corrected policy gradient with discrete bias correction
       Term 1 (truncated IS):
         −c_t · (Q^ret_t − V(s_t)) · ∇ log π(a_t|s_t)
       Term 2 (bias correction over all actions):
         −Σ_a max(0, π(a|s_t) − c_bar·μ(a|s_t)) · (Q(s_t,a) − V(s_t)) · ∇ log π(a|s_t)

  3. EMA average-policy trust region (KL penalty)
     Keeps the online policy from moving too far from the slowly-evolving
     average policy π_avg.  π_avg is updated after every gradient step:
       π_avg ← ema_alpha·π_avg + (1 − ema_alpha)·π_θ

  4. Replay buffer of complete episodes
     After each on-policy episode:
       • push episode to buffer
       • one on-policy gradient step
       • acer_replay_ratio off-policy gradient steps from random buffer episodes

Interface (mirrors A2C / PPO):
  select(state)              — sample action, cache μ(·|s) for replay
  record_transition(r, done) — attach reward and done to last transition
  update()                   — on-policy + off-policy gradient steps + EMA
  predict(state)             — greedy action for inference
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
from models import Encoder
from env import VOCAB_SIZE


# ── Network ───────────────────────────────────────────────────────────────────

class ACERNet(nn.Module):
    """Shared Transformer encoder with an actor head and a Q-value head.

    Unlike ActorCriticNet (which outputs logits + V(s)), ACERNet outputs:
      • logits  — unnormalised scores for each action (→ π via softmax)
      • q_all   — Q(s, a) for *every* action simultaneously
      • v       — V(s) = E_π[Q(s,a)] = (π · q_all).sum(dim=-1)

    Having Q(s,a) for all actions in one forward pass enables:
      (a) RETRACE bias-correction sum Σ_a over all actions,
      (b) advantage estimates A(s,a) = Q(s,a) − V(s) for every action.
    """

    def __init__(self, seq_num: int, max_seq_len: int,
                 action_number: int, d_model: int = 64):
        super().__init__()
        dim  = seq_num * (max_seq_len + 1)
        flat = dim * d_model
        self.encoder = Encoder(VOCAB_SIZE, d_model, dim)

        # Actor head — logits over actions
        self.actor = nn.Sequential(
            nn.Linear(flat, 512),
            nn.LeakyReLU(),
            nn.Linear(512, action_number),
        )
        # Q-head — Q(s, a) for all actions
        self.q_head = nn.Sequential(
            nn.Linear(flat, 512),
            nn.LeakyReLU(),
            nn.Linear(512, action_number),
        )

    def forward(self, x):
        """(B, state_dim) → (logits (B,A), q_all (B,A), v (B,))."""
        enc    = self.encoder(x, (x != 0).unsqueeze(-2))
        enc    = enc.reshape(enc.size(0), -1)
        logits = self.actor(enc)                         # (B, A)
        q_all  = self.q_head(enc)                        # (B, A)
        pi     = F.softmax(logits, dim=-1)               # (B, A)
        v      = (pi * q_all).sum(dim=-1)                # (B,)  V = E_π[Q]
        return logits, q_all, v

    def act(self, x):
        """Sample one action; return everything needed for the replay buffer.

        Returns:
            action   (B,)   — sampled action index
            log_pi   (B, A) — log π(·|s) for ALL actions (stored as μ in buffer)
            pi       (B, A) — π(·|s)
            v        (B,)   — V(s)
            q_all    (B, A) — Q(s, a) for all a
        """
        logits, q_all, v = self.forward(x)
        log_pi = F.log_softmax(logits, dim=-1)           # (B, A)
        pi     = torch.exp(log_pi)                        # (B, A)
        action = torch.distributions.Categorical(probs=pi).sample()
        return action, log_pi, pi, v, q_all

    def predict(self, x):
        """Greedy (deterministic) action for inference."""
        logits, _, _ = self.forward(x)
        return torch.argmax(logits, dim=-1)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class EpisodeBuffer:
    """Fixed-capacity circular buffer that stores complete episodes.

    Each episode is a list of five-element records:
        [state, action, reward, done, log_mu_all]
    where ``log_mu_all`` is the log-softmax vector over ALL actions under the
    behaviour policy μ (= the online policy π at the moment of selection).
    Storing the full distribution enables exact discrete bias correction.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: list = []
        self._pos: int  = 0

    def push(self, episode: list) -> None:
        if len(self._buf) < self.capacity:
            self._buf.append(episode)
        else:
            self._buf[self._pos] = episode
            self._pos = (self._pos + 1) % self.capacity

    def sample(self, n: int) -> list:
        idx = np.random.choice(len(self._buf),
                               size=min(n, len(self._buf)),
                               replace=False)
        return [self._buf[i] for i in idx]

    def __len__(self) -> int:
        return len(self._buf)


# ── ACER agent ────────────────────────────────────────────────────────────────

class ACER:
    """Actor-Critic with Experience Replay (discrete action space).

    Key differences vs A2C
    ──────────────────────
    • Maintains a replay buffer of complete episodes — allows off-policy
      learning from past behaviour without storing individual transitions.
    • Uses Retrace(λ) targets instead of plain Monte-Carlo returns, giving
      lower-variance Q estimates with guaranteed convergence for any μ/π ratio.
    • Adds a full bias-correction term over all actions so the policy gradient
      is asymptotically unbiased despite IS truncation.
    • EMA average policy tracks a slow-moving copy of π; KL divergence from
      the average acts as a trust-region penalty preventing large policy jumps.
    • Off-policy replay (replay_ratio updates per on-policy step) increases
      sample efficiency at minimal per-step wall-clock cost.
    """

    def __init__(self, action_number: int, seq_num: int,
                 max_seq_len: int, d_model: int = 64):
        self.action_number = action_number

        # Online (learning) network
        self.net = ACERNet(
            seq_num, max_seq_len, action_number, d_model
        ).to(config.device)

        # EMA average network — params updated slowly; no gradients needed
        self.avg_net = ACERNet(
            seq_num, max_seq_len, action_number, d_model
        ).to(config.device)
        self.avg_net.load_state_dict(self.net.state_dict())
        for p in self.avg_net.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=config.acer_lr)
        self._use_amp = torch.cuda.is_available()
        self.scaler   = torch.amp.GradScaler("cuda", enabled=self._use_amp)

        self.replay = EpisodeBuffer(config.acer_replay_size)

        # Per-episode accumulator: [state, action, reward, done, log_mu_all]
        self._ep: list = []

    # ── Public interface (mirrors A2C / PPO) ──────────────────────────────────

    def select(self, state) -> int:
        """Sample an action and store the full behaviour log-distribution."""
        x = torch.LongTensor(state).unsqueeze(0).to(config.device)
        with torch.no_grad():
            action, log_mu, _, _, _ = self.net.act(x)
        a         = action.item()
        log_mu_np = log_mu.squeeze(0).cpu().numpy()   # shape (A,) — full μ dist
        self._ep.append([state, a, None, None, log_mu_np])
        return a

    def record_transition(self, reward: float, done: float) -> None:
        """Attach reward and done flag to the most recently selected action."""
        self._ep[-1][2] = reward
        self._ep[-1][3] = done

    def predict(self, state) -> int:
        """Greedy action for inference (no buffer update)."""
        x = torch.LongTensor(state).unsqueeze(0).to(config.device)
        with torch.no_grad():
            return self.net.predict(x).item()

    def update(self) -> None:
        """One on-policy gradient step + replay_ratio off-policy steps + EMA update."""
        if not self._ep:
            return

        ep = list(self._ep)
        self._ep.clear()
        self.replay.push(ep)

        # On-policy update on the episode just collected
        self._gradient_step(ep)

        # Off-policy updates from random buffer episodes
        if len(self.replay) > 1:
            for _ in range(config.acer_replay_ratio):
                self._gradient_step(self.replay.sample(1)[0])

        # Update EMA average policy
        ema = config.acer_ema_alpha
        with torch.no_grad():
            for p_avg, p_cur in zip(self.avg_net.parameters(),
                                    self.net.parameters()):
                p_avg.data.mul_(ema).add_((1.0 - ema) * p_cur.data)

    # ── Core gradient step ────────────────────────────────────────────────────

    def _gradient_step(self, episode: list) -> None:
        """One ACER gradient update from a single stored episode.

        Algorithm sketch
        ────────────────
        1. Forward pass through online and average networks.
        2. Compute IS ratios ρ_t = π(a_t|s_t) / μ(a_t|s_t) and c_t = min(c̄, ρ_t).
        3. RETRACE backward recursion → Q^ret targets.
        4. Policy loss  = truncated-IS PG + bias-correction PG.
        5. Critic loss  = Huber(Q(s_t,a_t), Q^ret_t).
        6. Entropy bonus.
        7. Trust-region KL penalty KL(π_avg ∥ π_θ).
        """
        T = len(episode)
        if T == 0:
            return

        # ── Batch assembly ────────────────────────────────────────────────────
        states  = torch.LongTensor(
            [e[0] for e in episode]).to(config.device)              # (T, state_dim)
        actions = torch.LongTensor(
            [e[1] for e in episode]).to(config.device)              # (T,)
        rewards = torch.FloatTensor(
            [e[2] for e in episode]).to(config.device)              # (T,)

        # ── Per-pair reward normalisation ─────────────────────────────────────
        # SP reward = sum over C(seq_num,2) pairs → scales as O(seq_num²).
        # Dividing by n_pairs keeps reward magnitude ≈ ±4 regardless of how
        # many sequences are in the dataset, so acer_entropy_coef is comparable
        # across 3×30 bp (3 pairs) and 6×30 bp / 6×60 bp (15 pairs).
        _seq_num = int(math.log2(self.action_number + 1))
        _n_pairs = max(1, _seq_num * (_seq_num - 1) // 2)
        rewards  = rewards / _n_pairs

        dones   = torch.FloatTensor(
            [e[3] for e in episode]).to(config.device)              # (T,)
        log_mu  = torch.FloatTensor(                                # (T, A)
            np.stack([e[4] for e in episode])
        ).to(config.device)

        idx_t = torch.arange(T, device=config.device)

        with torch.amp.autocast("cuda", enabled=self._use_amp):

            # ── Forward: online network ───────────────────────────────────────
            logits, q_all, v = self.net(states)         # (T,A), (T,A), (T,)
            log_pi = F.log_softmax(logits, dim=-1)       # (T, A)
            pi     = torch.exp(log_pi)                   # (T, A)

            # ── Forward: average network (detached — no gradients) ────────────
            with torch.no_grad():
                avg_logits, _, _ = self.avg_net(states)
                avg_log_pi = F.log_softmax(avg_logits, dim=-1)   # (T, A)
                avg_pi     = torch.exp(avg_log_pi)                # (T, A)

            # ── Importance sampling ratios ────────────────────────────────────
            # ρ_t = π(a_t|s_t) / μ(a_t|s_t)  for the taken action
            log_rho = (log_pi[idx_t, actions]
                       - log_mu[idx_t, actions]).detach()          # (T,)
            rho     = log_rho.exp()                                 # (T,)
            c       = rho.clamp(max=config.acer_c_bar)             # (T,) c_t

            # ── RETRACE Q^ret targets (backward recursion) ───────────────────
            # Q^ret_t = r_t + γ·done_t·(V(s_{t+1}) + c_{t+1}·(Q^ret_{t+1} − Q(s_{t+1},a_{t+1})))
            # done=0 ⟹ terminal ⟹ bootstrap = 0  (last step always done=0)
            with torch.no_grad():
                q_all_d = q_all.detach()                           # (T, A)
                v_d     = v.detach()                               # (T,)
                q_taken = q_all_d[idx_t, actions]                  # (T,)

                Q_ret = torch.zeros(T, device=config.device)
                for t in reversed(range(T)):
                    if t == T - 1:
                        # Terminal: V(s_T) = 0
                        Q_ret[t] = rewards[t]
                    else:
                        Q_ret[t] = (
                            rewards[t]
                            + config.gamma * dones[t]
                            * (v_d[t + 1]
                               + c[t + 1] * (Q_ret[t + 1] - q_taken[t + 1]))
                        )

            # ── Policy gradient ───────────────────────────────────────────────
            adv = (Q_ret - v_d).detach()                           # (T,)  A^ret

            # Term 1: truncated IS policy gradient (c_t ≤ c_bar)
            log_pi_taken = log_pi[idx_t, actions]                  # (T,)
            pg1 = -(c.detach() * adv * log_pi_taken)               # (T,)

            # Term 2: bias correction — sum over all actions
            # coeff[a] = max(0, π(a) − c_bar·μ(a))
            mu       = torch.exp(log_mu).detach()                  # (T, A)
            bc_coeff = (pi.detach()
                        - config.acer_c_bar * mu).clamp(min=0.0)   # (T, A)
            adv_all  = (q_all.detach()
                        - v_d.unsqueeze(-1))                        # (T, A)
            pg2      = -(bc_coeff * adv_all * log_pi).sum(dim=-1)  # (T,)

            policy_loss = (pg1 + pg2).mean()

            # ── Critic loss ───────────────────────────────────────────────────
            # Huber(Q(s_t, a_t), Q^ret_t) — gradient through q_all only
            q_taken_cur = q_all[idx_t, actions]                    # (T,)
            critic_loss = F.smooth_l1_loss(q_taken_cur, Q_ret.detach())

            # ── Entropy bonus (higher = more exploratory policy) ─────────────
            entropy = -(pi * log_pi).sum(dim=-1).mean()            # scalar ≥ 0

            # ── Trust-region KL penalty: KL(π_avg ∥ π_θ) ────────────────────
            # = Σ_a π_avg·(log π_avg − log π) ; gradient flows through log_pi
            kl = (avg_pi.detach()
                  * (avg_log_pi.detach() - log_pi)
                  ).sum(dim=-1).clamp(min=0.0).mean()

            # ── Total loss ────────────────────────────────────────────────────
            total_loss = (
                policy_loss
                + config.acer_value_coef         * critic_loss
                - config.acer_entropy_coef        * entropy
                + config.acer_trust_region_delta  * kl
            )

        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, filename: str, path: str = config.weight_path) -> None:
        torch.save(self.net.state_dict(),
                   os.path.join(path, f"{filename}.pth"))
        print(f"{filename} has been saved...")

    def load(self, filename: str, path: str = config.weight_path) -> None:
        self.net.load_state_dict(
            torch.load(os.path.join(path, f"{filename}.pth"),
                       map_location=config.device))
        # Sync average network so trust region starts centred on the loaded policy
        self.avg_net.load_state_dict(self.net.state_dict())
        print(f"{filename} has been loaded...")
