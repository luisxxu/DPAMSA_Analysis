import os.path
import platform
import torch
import math

GAP_PENALTY = -4
MISMATCH_PENALTY = -4
MATCH_REWARD = 4

update_iteration = 128

batch_size = 256          # Increased from 128: better GPU utilization
# Altered line of code: Max_episode parameter to test its effects
max_episode = 120
replay_memory_size = 10000  # Increased from 1000: more diverse experience, reduces catastrophic forgetting

alpha = 0.0001
gamma = 0.99              # Changed from 1: discounting stabilizes training and improves convergence
epsilon = 0.9             # Increased from 0.8: more exploration early in training
delta = 0.05

decrement_iteration = math.ceil(max_episode * 0.8 / (epsilon // delta))

device_name = "cuda:1" if torch.cuda.is_available() else "cpu"

weight_path = "../result/weight"
score_path = "../result/score"
report_path = "../result/report"

if not os.path.exists(score_path):
    os.makedirs(score_path)
if not os.path.exists(weight_path):
    os.makedirs(weight_path)
if not os.path.exists(report_path):
    os.makedirs(report_path)

assert 0 < batch_size <= replay_memory_size, "batch size must be in the range of 0 to the size of replay memory."
assert alpha > 0, "alpha must be greater than 0."
assert 0 <= gamma <= 1, "gamma must be in the range of 0 to 1."
assert 0 <= epsilon <= 1, "epsilon must be in the range of 0 to 1."
assert 0 <= delta <= epsilon, "delta must be in the range of 0 to epsilon."
assert 0 < decrement_iteration, "decrement iteration must be greater than 0."

# ---------------------------------------------------------------------------
# A2C (Advantage Actor-Critic) hyperparameters
# ---------------------------------------------------------------------------
a2c_lr           = 0.0003  # Learning rate (Adam). Higher than DQN's alpha because
                            # on-policy gradients are lower-variance and tolerate
                            # larger steps.
a2c_value_coef   = 0.5     # Weight of the critic (value) loss term.
a2c_entropy_coef = 0.01    # Weight of the entropy bonus. Encourages exploration
                            # by penalising a deterministic policy.

# ---------------------------------------------------------------------------
# PPO (Proximal Policy Optimisation) hyperparameters
# ---------------------------------------------------------------------------
ppo_lr           = 0.0003  # Learning rate for PPO (same scale as A2C).
ppo_clip_eps     = 0.2     # Clipping range for the probability ratio r_t.
                            # Constrains π_new/π_old to [0.8, 1.2].
ppo_epochs       = 4       # Number of gradient update epochs per rollout.
                            # Higher values reuse data more but risk overfitting
                            # the old-policy data.
ppo_value_coef   = 0.5     # Weight of the critic loss term.
ppo_entropy_coef = 0.01    # Entropy bonus coefficient (same purpose as A2C).
gae_lambda       = 0.95    # GAE λ — smoothly interpolates between TD(0) (λ=0)
                            # and Monte-Carlo returns (λ=1).

assert 0 < a2c_lr,           "A2C learning rate must be positive."
assert 0 < ppo_lr,           "PPO learning rate must be positive."
assert 0 < ppo_clip_eps < 1, "PPO clip epsilon must be in (0, 1)."
assert 0 < ppo_epochs,       "PPO epochs must be at least 1."
assert 0 <= gae_lambda <= 1, "GAE lambda must be in [0, 1]."
