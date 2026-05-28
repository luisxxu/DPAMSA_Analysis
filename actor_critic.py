from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
import os
from models import Encoder
from env import VOCAB_SIZE


class ActorCriticNet(nn.Module):
    """Shared Transformer encoder with separate actor and critic heads.

    Used by both A2C and PPO. The encoder (imported from models.py) extracts
    sequence representations; the actor head produces a categorical distribution
    over gap-insertion actions; the critic head estimates the state value V(s).

    Having a shared backbone means both heads benefit from the same learned
    sequence features, while the separate heads allow them to specialise
    independently.
    """

    def __init__(self, seq_num, max_seq_len, action_number, d_model=64):
        super().__init__()
        dim = seq_num * (max_seq_len + 1)
        self.encoder = Encoder(VOCAB_SIZE, d_model, dim)

        # Actor head — outputs un-normalised logits, one per action
        self.actor = nn.Sequential(
            nn.Linear(dim * d_model, 512),
            nn.LeakyReLU(),
            nn.Linear(512, action_number),
        )

        # Critic head — outputs a single scalar V(s)
        self.critic = nn.Sequential(
            nn.Linear(dim * d_model, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        enc    = self.encoder(x, (x != 0).unsqueeze(-2))
        enc    = enc.reshape(enc.size(0), -1)
        logits = self.actor(enc)
        value  = self.critic(enc).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x):
        """Sample one action from π(·|s); return (action, log_prob, value, entropy)."""
        logits, value = self.forward(x)
        dist          = torch.distributions.Categorical(logits=logits)
        action        = dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()

    def evaluate_actions(self, x, actions):
        """Re-evaluate stored actions under the current policy (used during updates)."""
        logits, value = self.forward(x)
        dist          = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), value, dist.entropy()

    def predict(self, x):
        """Greedy (deterministic) action for inference — no sampling."""
        logits, _ = self.forward(x)
        return torch.argmax(logits, dim=-1)


class ActorCritic(ABC):
    """Advantage Actor-Critic (A2C) agent.

    On-policy algorithm that collects a complete episode, then performs a
    single gradient update using one-step TD advantage estimates:

        A(s_t, a_t) = R_t − V(s_t)
        R_t         = r_t + γ · done_t · R_{t+1}   (done=1 → continue, 0 → terminal)

    Total loss:
        L = −E[log π(a|s) · A]           (policy gradient, maximise advantage)
          + value_coef  · Huber(V(s), R) (critic regression)
          − entropy_coef · H[π(·|s)]     (entropy bonus for exploration)

    Key differences vs DQN
    ───────────────────────
    • No experience replay — on-policy, data is discarded after each update.
    • Explicit stochastic policy rather than ε-greedy action selection.
    • Critic baseline reduces variance without introducing DQN's target-net lag.
    • Entropy regularisation replaces ε-decay for managing exploration.
    """

    def __init__(self, action_number, seq_num, max_seq_len):
        super().__init__()
        self.action_number = action_number
        self.net = ActorCriticNet(seq_num, max_seq_len, action_number).to(config.device)

        try:
            self.net = torch.compile(self.net)
        except Exception:
            pass  # torch.compile not available on PyTorch < 2.0

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.a2c_lr)
        self._use_amp  = torch.cuda.is_available()
        self.scaler    = torch.cuda.amp.GradScaler(enabled=self._use_amp)

        # Per-episode trajectory buffer — cleared at every update
        self._states  = []
        self._actions = []
        self._rewards = []
        self._values  = []
        self._dones   = []

    # ------------------------------------------------------------------
    # Interface (mirrors DQN.select / DQN.predict)
    # ------------------------------------------------------------------

    def select(self, state):
        """Sample action from π(·|s); cache state, action, and value estimate."""
        x = torch.LongTensor(state).unsqueeze(0).to(config.device)
        with torch.no_grad():
            action, _, value, _ = self.net.get_action_and_value(x)
        a = action.item()
        self._states.append(state)
        self._actions.append(a)
        self._values.append(value.item())
        return a

    def record_transition(self, reward: float, done: float):
        """Record the (reward, done) outcome of the last selected action.

        done=1.0 means 'continue'; done=0.0 means 'terminal' — matching the
        convention used throughout this codebase (opposite to gym's done=True).
        """
        self._rewards.append(reward)
        self._dones.append(done)

    def predict(self, state):
        """Greedy action for inference; no sampling and no buffer update."""
        x = torch.LongTensor(state).unsqueeze(0).to(config.device)
        with torch.no_grad():
            return self.net.predict(x).item()

    # ------------------------------------------------------------------
    # Update (single environment)
    # ------------------------------------------------------------------

    def update(self):
        """Compute advantages over the collected episode and perform one gradient step."""
        if not self._rewards:
            return

        # Discounted returns — bootstrap value = 0 because every episode in
        # this environment terminates naturally (last done == 0.0).
        R       = 0.0
        returns = []
        for r, d in zip(reversed(self._rewards), reversed(self._dones)):
            R = r + config.gamma * d * R   # d=0 stops accumulation at terminal
            returns.insert(0, R)

        states_t  = torch.LongTensor(self._states).to(config.device)
        actions_t = torch.LongTensor(self._actions).to(config.device)
        returns_t = torch.FloatTensor(returns).to(config.device)
        adv_t     = returns_t - torch.FloatTensor(self._values).to(config.device)
        adv_t     = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        self._gradient_step(states_t, actions_t, returns_t, adv_t)

        # Discard trajectory — on-policy means no replay
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._values.clear()
        self._dones.clear()

    # ------------------------------------------------------------------
    # Update (parallel environments)
    # ------------------------------------------------------------------

    def update_parallel(self, traj_s, traj_a, traj_r, traj_v, traj_d):
        """Update from N parallel episode trajectories.

        Computes per-trajectory discounted returns, concatenates them into
        one batch, normalises advantages across all trajectories together,
        then performs a single gradient step — equivalent to A2C with a
        batch size N× larger than the single-env version.

        Args:
            traj_s/a/r/v/d : lists of N inner lists, one per environment.
                             Each inner list holds the steps from that env's episode.
        """
        all_states, all_actions, all_returns, all_adv = [], [], [], []

        for env_i in range(len(traj_r)):
            if not traj_r[env_i]:
                continue
            # Per-trajectory discounted returns
            R, returns = 0.0, []
            for r, d in zip(reversed(traj_r[env_i]), reversed(traj_d[env_i])):
                R = r + config.gamma * d * R
                returns.insert(0, R)
            adv = [ret - val for ret, val in zip(returns, traj_v[env_i])]
            all_states.extend(traj_s[env_i])
            all_actions.extend(traj_a[env_i])
            all_returns.extend(returns)
            all_adv.extend(adv)

        if not all_states:
            return

        states_t  = torch.LongTensor(all_states).to(config.device)
        actions_t = torch.LongTensor(all_actions).to(config.device)
        returns_t = torch.FloatTensor(all_returns).to(config.device)
        adv_t     = torch.FloatTensor(all_adv).to(config.device)
        # Normalise across ALL n_envs trajectories — more stable than per-trajectory
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        self._gradient_step(states_t, actions_t, returns_t, adv_t)

    # ------------------------------------------------------------------
    # Shared gradient step (used by both update variants)
    # ------------------------------------------------------------------

    def _gradient_step(self, states_t, actions_t, returns_t, adv_t):
        """One A2C gradient update on pre-assembled tensors."""
        with torch.cuda.amp.autocast(enabled=self._use_amp):
            log_probs, values, entropy = self.net.evaluate_actions(states_t, actions_t)

            actor_loss   = -(log_probs * adv_t).mean()
            critic_loss  = F.smooth_l1_loss(values, returns_t)
            entropy_loss = -entropy.mean()

            loss = (actor_loss
                    + config.a2c_value_coef   * critic_loss
                    + config.a2c_entropy_coef * entropy_loss)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filename, path=config.weight_path):
        torch.save(self.net.state_dict(), os.path.join(path, f"{filename}.pth"))
        print(f"{filename} has been saved...")

    def load(self, filename, path=config.weight_path):
        self.net.load_state_dict(torch.load(
            os.path.join(path, f"{filename}.pth"),
            map_location=config.device))
        print(f"{filename} has been loaded...")
