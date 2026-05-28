from abc import ABC
import torch
import torch.nn as nn
import numpy as np
import config
import random
import os
from models import Encoder
from replay_memory import ReplayMemory
# VOCAB_SIZE is defined alongside nucleotides_map in env.py so that the
# vocabulary size stays in sync automatically whenever new token IDs are added.
from env import VOCAB_SIZE


class Net(nn.Module):
    """DQN network: Transformer encoder followed by three fully-connected layers."""

    def __init__(self, seq_num, max_seq_len, action_number, max_value, d_model=64):
        super(Net, self).__init__()
        self.max_value = max_value
        dim = seq_num * (max_seq_len + 1)
        # Embedding improvement A/B: VOCAB_SIZE (imported from env.py) replaces
        # the hardcoded 6 so the encoder automatically tracks any vocabulary changes.
        self.encoder = Encoder(VOCAB_SIZE, d_model, dim)
        self.dropout = nn.Dropout()
        self.l1 = nn.Linear(dim * d_model, 1028)
        self.f1 = nn.LeakyReLU()
        self.l2 = nn.Linear(1028, 512)
        self.f2 = nn.LeakyReLU()
        self.l3 = nn.Linear(512, action_number)
        self.f3 = nn.Tanh()

        self.mask = lambda x, y: (x != y).unsqueeze(-2)

    def forward(self, x):
        x = self.encoder(x, self.mask(x, 0))
        x = x.reshape(x.size()[0], -1)
        x = self.f1(self.l1(x))
        x = self.f2(self.l2(x))
        x = self.f3(self.l3(x))
        x = torch.mul(x, self.max_value)
        return x


class DQN(ABC):
    def __init__(self, action_number, seq_num, max_seq_len, max_value):
        super(DQN, self).__init__()
        self.seq_num = seq_num
        self.max_seq_len = max_seq_len
        self.action_number = action_number
        self.eval_net = Net(seq_num, max_seq_len, action_number, max_value).to(config.device)
        self.target_net = Net(seq_num, max_seq_len, action_number, max_value).to(config.device)

        # Improvement 8: torch.compile() (PyTorch >= 2.0) JIT-compiles the model
        # graph, fusing operations and eliminating Python overhead for 10-30%
        # additional speedup at zero algorithmic cost. Falls back silently on
        # older PyTorch versions.
        try:
            self.eval_net = torch.compile(self.eval_net)
            self.target_net = torch.compile(self.target_net)
        except Exception:
            pass  # torch.compile not available (PyTorch < 2.0)

        self.current_epsilon = config.epsilon

        self.update_step_counter = 0
        self.epsilon_step_counter = 0

        self.replay_memory = ReplayMemory()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=config.alpha)

        # Improvement 2: Huber loss (SmoothL1) instead of MSE.
        # MSE heavily penalises large TD errors, causing unstable gradient spikes.
        # Huber loss is quadratic for small errors and linear for large ones,
        # giving stable gradients throughout training.
        self.loss_func = nn.SmoothL1Loss()

        # Improvement 3: Mixed-precision scaler for torch.cuda.amp.
        # enabled=False is a no-op on CPU so this is safe in all environments.
        self._use_amp = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)

    def update_epsilon(self):
        self.epsilon_step_counter += 1
        if self.epsilon_step_counter % config.decrement_iteration == 0:
            self.current_epsilon = max(0, self.current_epsilon - config.delta)

    def select(self, state):
        # random policy
        if random.random() <= self.current_epsilon:
            action = np.random.randint(0, self.action_number)
        # greedy policy
        else:
            action_val = self.eval_net.forward(
                torch.LongTensor(state).unsqueeze(0).to(config.device))
            action = torch.argmax(action_val, 1).cpu().data.numpy()[0]
        return action

    def select_batch(self, states):
        """Batched epsilon-greedy for N states — one forward pass for all greedy actions.

        Random actions are sampled independently (no GPU needed). Greedy actions
        are evaluated together in a single (|greedy|, state_dim) forward pass,
        which is much faster than N individual forward passes when many states
        are in the greedy branch (i.e. late training when epsilon is low).

        Args:
            states: list of N state vectors (one per active environment).
        Returns:
            list of N integer actions.
        """
        n = len(states)
        # Decide random vs greedy for each env independently
        use_random = [random.random() <= self.current_epsilon for _ in range(n)]

        # Start with random actions everywhere
        actions = [int(np.random.randint(0, self.action_number)) for _ in range(n)]

        # Overwrite greedy slots with one batched forward pass
        greedy_idx = [i for i, r in enumerate(use_random) if not r]
        if greedy_idx:
            greedy_states = [states[i] for i in greedy_idx]
            x = torch.LongTensor(greedy_states).to(config.device)
            with torch.no_grad():
                action_vals = self.eval_net(x)
            greedy_actions = torch.argmax(action_vals, dim=1).cpu().numpy()
            for j, env_i in enumerate(greedy_idx):
                actions[env_i] = int(greedy_actions[j])

        return actions

    def predict(self, state):
        action_val = self.eval_net.forward(
            torch.LongTensor(state).unsqueeze(0).to(config.device))
        return torch.argmax(action_val, 1).cpu().data.numpy()[0]

    def update(self):
        # Sync target network periodically
        self.update_step_counter += 1
        if self.update_step_counter % config.update_iteration == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        if self.replay_memory.size < config.batch_size:
            return

        # Improvement 6: sample returns indices so priorities can be updated
        state, next_state, action, reward, done, indices = \
            self.replay_memory.sample(config.batch_size)

        batch_state      = torch.LongTensor(state).to(config.device)
        batch_next_state = torch.LongTensor(next_state).to(config.device)
        batch_action     = torch.LongTensor(action).to(config.device)
        batch_reward     = torch.FloatTensor(reward).to(config.device)
        batch_done       = torch.FloatTensor(done).to(config.device)

        # Improvement 3: Mixed-precision forward passes — ~2x faster on GPU.
        # autocast is a no-op when enabled=False (CPU), so this is always safe.
        with torch.cuda.amp.autocast(enabled=self._use_amp):
            # q_eval: Q-value of the action actually taken
            q_eval = self.eval_net(batch_state).gather(
                1, batch_action.unsqueeze(1)).squeeze(1)

            # Improvement 5: Double DQN target computation.
            # Vanilla DQN uses the target_net for both action selection and
            # evaluation, causing systematic overestimation of Q-values.
            # Double DQN uses eval_net to SELECT the best next action and
            # target_net to EVALUATE it, decoupling the two and removing the
            # maximisation bias.
            with torch.no_grad():
                best_next_actions = self.eval_net(batch_next_state).argmax(
                    1, keepdim=True)
                q_next = self.target_net(batch_next_state).gather(
                    1, best_next_actions).squeeze(1).detach()

            # config.gamma = 0.99 (was 1): discounting stabilises training
            q_target = batch_reward + batch_done * config.gamma * q_next

            # Improvement 2: SmoothL1Loss (Huber)
            loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        # Improvement 2: Gradient clipping — unscale first so the clip
        # threshold is in the original gradient units, not scaled units.
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=10.0)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Improvement 6: update replay priorities with absolute TD errors
        with torch.no_grad():
            td_errors = (q_eval.detach().float() - q_target.float()).abs().cpu().numpy()
        self.replay_memory.update_priorities(indices, td_errors)

    def save(self, filename, path=config.weight_path):
        torch.save(self.eval_net.state_dict(), os.path.join(path, "{}.pth".format(filename)))
        print("{} has been saved...".format(filename))

    def load(self, filename, path=config.weight_path):
        self.eval_net.load_state_dict(torch.load(os.path.join(path, "{}.pth".format(filename)),
                                                 map_location=torch.device(config.device)))
        print("{} has been loaded...".format(filename))
