from abc import ABC
import torch
import torch.nn.functional as F
import config
import os
# ActorCriticNet is shared between A2C and PPO — both use the same
# encoder + actor-head + critic-head architecture.
from actor_critic import ActorCriticNet


class PPO(ABC):
    """Proximal Policy Optimisation (PPO-Clip) agent.

    PPO extends A2C with three improvements that together make it one of
    the most reliable and sample-efficient deep RL algorithms in practice:

    1. Generalised Advantage Estimation (GAE, Schulman et al. 2015)
       ─────────────────────────────────────────────────────────────
       Instead of one-step TD advantages, GAE computes an exponentially-
       weighted average of k-step returns, controlled by λ ∈ [0,1]:

           δ_t  = r_t + γ · done_t · V(s_{t+1}) − V(s_t)
           Â_t  = δ_t + (γλ) · done_t · Â_{t+1}

       λ=0 → pure TD(0) (low variance, high bias)
       λ=1 → Monte-Carlo (high variance, zero bias)
       λ=0.95 (default) gives a good bias-variance trade-off.

    2. Clipped surrogate objective
       ────────────────────────────
       The probability ratio r_t = π_new(a|s) / π_old(a|s) measures how
       much the policy has changed.  Clipping it to [1−ε, 1+ε] prevents
       the update from making the policy change too drastically in one step:

           L_CLIP = E[min(r_t · Â_t,  clip(r_t, 1−ε, 1+ε) · Â_t)]

    3. Multiple update epochs per rollout
       ────────────────────────────────────
       After collecting one episode, PPO runs K gradient steps on the same
       batch (default K=4), improving sample efficiency over A2C's single step.
       The clip constraint keeps successive updates safe.

    Total loss per epoch:
        L = −L_CLIP
          + value_coef  · Huber(V(s), returns)
          − entropy_coef · H[π(·|s)]
    """

    def __init__(self, action_number, seq_num, max_seq_len):
        super().__init__()
        self.action_number = action_number
        self.net = ActorCriticNet(seq_num, max_seq_len, action_number).to(config.device)

        try:
            self.net = torch.compile(self.net)
        except Exception:
            pass  # torch.compile not available on PyTorch < 2.0

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.ppo_lr)
        self._use_amp  = torch.cuda.is_available()
        self.scaler    = torch.cuda.amp.GradScaler(enabled=self._use_amp)

        # Rollout buffer — cleared after every update
        self._states    = []
        self._actions   = []
        self._log_probs = []   # log π_old(a|s) — the policy that collected the data
        self._rewards   = []
        self._values    = []
        self._dones     = []

    # ------------------------------------------------------------------
    # Interface (mirrors DQN.select / DQN.predict)
    # ------------------------------------------------------------------

    def select(self, state):
        """Sample action from π(·|s); cache state, action, log-prob, and value."""
        x = torch.LongTensor(state).unsqueeze(0).to(config.device)
        with torch.no_grad():
            action, log_prob, value, _ = self.net.get_action_and_value(x)
        a = action.item()
        self._states.append(state)
        self._actions.append(a)
        self._log_probs.append(log_prob.item())
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
        """Run ppo_epochs gradient steps on the collected rollout."""
        if not self._rewards:
            return

        advantages, returns = self._gae_for_traj(
            self._rewards, self._values, self._dones)

        states_t  = torch.LongTensor(self._states).to(config.device)
        actions_t = torch.LongTensor(self._actions).to(config.device)
        old_lp_t  = torch.FloatTensor(self._log_probs).to(config.device)
        returns_t = torch.FloatTensor(returns).to(config.device)
        adv_t     = torch.FloatTensor(advantages).to(config.device)
        adv_t     = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        self._gradient_steps(states_t, actions_t, old_lp_t, returns_t, adv_t)

        # Clear rollout buffer — the old-policy log-probs are now stale
        self._states.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._rewards.clear()
        self._values.clear()
        self._dones.clear()

    # ------------------------------------------------------------------
    # Update (parallel environments)
    # ------------------------------------------------------------------

    def update_parallel(self, traj_s, traj_a, traj_lp, traj_r, traj_v, traj_d):
        """Update from N parallel episode trajectories.

        Computes per-trajectory GAE, concatenates into one large batch,
        normalises advantages across all N trajectories, then runs
        ppo_epochs clipped updates — equivalent to PPO on a rollout N×
        the size of the single-env version.

        Args:
            traj_s/a/lp/r/v/d : lists of N inner lists, one per environment.
                                 traj_lp holds log π_old(a|s) from collection.
        """
        all_states, all_actions, all_log_probs = [], [], []
        all_returns, all_adv = [], []

        for env_i in range(len(traj_r)):
            if not traj_r[env_i]:
                continue
            adv, returns = self._gae_for_traj(
                traj_r[env_i], traj_v[env_i], traj_d[env_i])
            all_states.extend(traj_s[env_i])
            all_actions.extend(traj_a[env_i])
            all_log_probs.extend(traj_lp[env_i])
            all_returns.extend(returns)
            all_adv.extend(adv)

        if not all_states:
            return

        states_t  = torch.LongTensor(all_states).to(config.device)
        actions_t = torch.LongTensor(all_actions).to(config.device)
        old_lp_t  = torch.FloatTensor(all_log_probs).to(config.device)
        returns_t = torch.FloatTensor(all_returns).to(config.device)
        adv_t     = torch.FloatTensor(all_adv).to(config.device)
        adv_t     = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        self._gradient_steps(states_t, actions_t, old_lp_t, returns_t, adv_t)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gae_for_traj(rewards, values, dones):
        """GAE for a single trajectory.

        δ_t  = r_t + γ · done_t · V(s_{t+1}) − V(s_t)
        Â_t  = δ_t + γλ · done_t · Â_{t+1}

        Bootstrap value = 0 (episodes always terminate naturally, last done==0).
        done=1.0 → discount future; done=0.0 → terminal, stop accumulation.
        """
        advantages      = []
        gae             = 0.0
        values_extended = list(values) + [0.0]

        for t in reversed(range(len(rewards))):
            d     = dones[t]
            delta = (rewards[t]
                     + config.gamma * d * values_extended[t + 1]
                     - values_extended[t])
            gae   = delta + config.gamma * config.gae_lambda * d * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def _gradient_steps(self, states_t, actions_t, old_lp_t, returns_t, adv_t):
        """Run ppo_epochs clipped PPO updates on pre-assembled tensors."""
        for _ in range(config.ppo_epochs):
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                new_log_probs, values, entropy = self.net.evaluate_actions(
                    states_t, actions_t)

                # Probability ratio π_new / π_old (log-space for stability)
                ratio = torch.exp(new_log_probs - old_lp_t)

                surr_unclipped = ratio * adv_t
                surr_clipped   = torch.clamp(
                    ratio,
                    1.0 - config.ppo_clip_eps,
                    1.0 + config.ppo_clip_eps) * adv_t
                policy_loss  = -torch.min(surr_unclipped, surr_clipped).mean()
                value_loss   = F.smooth_l1_loss(values, returns_t)
                entropy_loss = -entropy.mean()

                loss = (policy_loss
                        + config.ppo_value_coef   * value_loss
                        + config.ppo_entropy_coef * entropy_loss)

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
