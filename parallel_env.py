from env import Environment


class ParallelEnvironment:
    """Runs n_envs independent Environment instances in lockstep.

    Why this speeds training up
    ───────────────────────────
    The training bottleneck for on-policy algorithms (A2C, PPO) is not the
    gradient update — it is data *collection*.  With a single environment,
    every forward pass processes exactly one state vector.  Modern GPUs can
    process a batch of N state vectors in almost the same wall-clock time as
    one, so running N environments simultaneously and stacking their states
    into a single (N, state_dim) tensor gives roughly N× throughput on the
    network at negligible extra cost.

    For DQN the gain is different: N envs fill the replay buffer N× faster,
    meaning the agent reaches the minimum batch size sooner and begins
    learning from more diverse experience earlier.

    Usage (from a training loop)
    ────────────────────────────
    par_env = ParallelEnvironment(sequences, n_envs=4)
    states  = par_env.reset_all()           # list of 4 states

    while not par_env.all_done():
        active_idx    = [i for i, a in enumerate(par_env.active_mask) if a]
        active_states = [states[i] for i in active_idx]

        # ONE forward pass on a (|active|, state_dim) tensor
        actions = batched_select(active_states)

        full_actions = [0] * par_env.n_envs
        for j, env_i in enumerate(active_idx):
            full_actions[env_i] = actions[j]

        results = par_env.step_all(full_actions)   # list of n_envs results

        for j, env_i in enumerate(active_idx):
            reward, next_state, done = results[env_i]
            if done != 0:
                states[env_i] = next_state

    Notes
    ─────
    • All envs are initialised on the same sequences so action_number, row,
      max_len, and max_reward are identical across every instance.
    • Inactive environments (done==0) return (0.0, None, 0) from step_all()
      so callers can safely iterate over the full n_envs result list.
    • After training, par_env.envs[0] is used for greedy inference and score
      reporting (its state is consistent because it ran a full episode).
    """

    def __init__(self, sequences: list, n_envs: int):
        self.n_envs = n_envs
        self.envs   = [Environment(sequences) for _ in range(n_envs)]
        self._active = [True] * n_envs

        # All envs share these since they run the same sequences
        self.action_number = self.envs[0].action_number
        self.row           = self.envs[0].row
        self.max_len       = self.envs[0].max_len
        self.max_reward    = self.envs[0].max_reward

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset_all(self) -> list:
        """Reset every environment and return a list of N initial states."""
        self._active = [True] * self.n_envs
        return [env.reset() for env in self.envs]

    def step_all(self, actions: list) -> list:
        """Step every *active* environment with the corresponding action.

        Returns a list of n_envs tuples: (reward, next_state, done).
        Inactive environments return (0.0, None, 0) — callers should check
        active_mask or skip indices not in their active_idx list.
        When an active env's done==0 it is automatically marked inactive.
        """
        results = []
        for i, env in enumerate(self.envs):
            if self._active[i]:
                reward, next_state, done = env.step(int(actions[i]))
                if done == 0:
                    self._active[i] = False
                results.append((reward, next_state, done))
            else:
                results.append((0.0, None, 0))
        return results

    @property
    def active_mask(self) -> list:
        """Boolean list: True for envs still running this episode."""
        return list(self._active)

    def all_done(self) -> bool:
        """True when every environment has finished its current episode."""
        return not any(self._active)

    # ------------------------------------------------------------------
    # Reporting interface — delegates to envs[0] (used after inference)
    # ------------------------------------------------------------------

    def padding(self):
        self.envs[0].padding()

    def get_alignment(self):
        return self.envs[0].get_alignment()

    def calc_score(self):
        return self.envs[0].calc_score()

    def calc_exact_matched(self):
        return self.envs[0].calc_exact_matched()

    @property
    def aligned(self):
        return self.envs[0].aligned
