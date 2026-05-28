import config
import numpy as np

"""
Improvement 6: Prioritized Experience Replay (PER).

The original implementation sampled transitions uniformly at random, meaning
all past experiences were treated as equally valuable. PER instead assigns
each transition a priority proportional to its TD error (how "surprising" it
was). Transitions the agent has not yet learned well are sampled more often,
improving sample efficiency and convergence speed.

Key design choices:
  - alpha (0.6): controls how strongly priorities are used
      0 = uniform sampling, 1 = fully greedy priority sampling
  - New transitions receive the current maximum priority so they are
      guaranteed to be sampled at least once before being deprioritised.
  - A small epsilon (1e-6) is added to every priority to ensure no
      transition has zero probability of being sampled.
  - update_priorities() is called from DQN.update() after each learning
      step with the absolute TD errors of the sampled batch.
"""


class ReplayMemory:
    def __init__(self):
        self.storage = []
        self.priorities = []
        self.max_size = config.replay_memory_size
        self.size = 0
        self.ptr = 0
        self.alpha = 0.6   # priority exponent: 0 = uniform, 1 = full priority

    def push(self, data: tuple, priority: float = 1.0):
        """Store a transition. New entries receive max existing priority."""
        max_priority = max(self.priorities, default=priority)
        if len(self.storage) < self.max_size:
            self.storage.append(data)
            self.priorities.append(max_priority)
            self.size += 1
        else:
            self.storage[self.ptr] = data
            self.priorities[self.ptr] = max_priority
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        """Sample a batch weighted by priority. Returns samples and their indices."""
        probs = np.array(self.priorities, dtype=np.float32) ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.storage), batch_size, replace=False, p=probs)
        samples = [self.storage[i] for i in indices]
        state, next_state, action, reward, done = zip(*samples)
        return (list(state), list(next_state), list(action),
                list(reward), list(done), indices.tolist())

    def update_priorities(self, indices, td_errors):
        """Update priorities for a sampled batch using absolute TD errors."""
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = float(abs(err)) + 1e-6
