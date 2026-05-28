# DPAMSA

## Running DPAMSA
The "run_DPAMSA.sh" bash script is written to be submitted to slurm. DPAMSA runs faster on a GPU node accessed through CUDA, but can also be run on a CPU node when a GPU is not available (automatically configured based on GPU availability). This bash script requests a GPU node through CUDA along with the required amount of time. The "config.py" script is to be editted when altering the parameters for DPAMSA. The code in "main.py" script was altered to accept fasta files containing the sequences to be aligned. Alter the bash script to specify the name of the fasta file that should be treated at the input for DPAMSA.

## Running traditional MSA techniques
The bash scripts for running ClustalW, T-COFFEE, and MAFFT are also designed to be submitted to slurm. Please check the report for information regarding how to download the traditional MSA techniques as well as the version numbers used for this analysis.

## Comparing the results
The "calc_scores.py" script is used to obtain the quality scores for the alignments. The scores can be used to compare the performances of the different MSA techniques.

## Performance Improvements
The original DPAMSA implementation had a significant runtime drawback noted in the report. The following changes were made across `env.py`, `models.py`, `replay_memory.py`, `dqn.py`, and `config.py` to reduce training time and improve convergence using modern deep RL techniques.

### 1. `deque.popleft()` instead of `list.pop(0)` — `env.py`
`self.not_aligned` was a plain Python list. `list.pop(0)` is O(n) because every remaining element must be shifted left after removal. Replacing it with `collections.deque` makes front removal O(1) via `popleft()`, which is called at every environment step and was a measurable CPU bottleneck for longer sequences.

### 2. Huber Loss + Gradient Clipping — `dqn.py`
`nn.MSELoss` was replaced with `nn.SmoothL1Loss` (Huber loss). MSE squares large TD errors, producing gradient spikes that destabilise training. Huber loss is quadratic for small errors and linear for large ones, keeping gradients bounded. Gradient clipping (`clip_grad_norm_`, max_norm=10.0) is applied after each backward pass for the same reason.

### 3. Mixed-Precision Training (`torch.cuda.amp`) — `dqn.py`
All forward passes in `update()` now run inside a `torch.cuda.amp.autocast` context with a `GradScaler`. This uses float16 for computations on GPU (where supported), roughly halving memory bandwidth usage and achieving ~2× speedup on modern CUDA hardware. The scaler is initialised with `enabled=torch.cuda.is_available()` so the code is a no-op on CPU and requires no conditional logic elsewhere.

### 4. Hyperparameter Tuning — `config.py`
| Parameter | Old value | New value | Reason |
|---|---|---|---|
| `gamma` | 1.0 | 0.99 | `gamma=1` (no discounting) causes Q-values to grow unboundedly and slows convergence; 0.99 stabilises training |
| `replay_memory_size` | 1 000 | 10 000 | Larger buffer stores more diverse experiences, reducing overfitting to recent transitions |
| `batch_size` | 128 | 256 | Larger batches improve GPU utilisation and produce lower-variance gradient estimates |
| `epsilon` | 0.8 | 0.9 | Higher initial exploration helps the agent discover better alignment strategies early |

### 5. Double DQN — `dqn.py`
Vanilla DQN uses the target network for both *selecting* and *evaluating* the best next action, which causes systematic Q-value overestimation (the "maximisation bias"). Double DQN decouples the two: the eval network selects the best next action (`argmax`) and the target network evaluates it (`gather`). This reduces overestimation bias and typically converges in fewer episodes.

### 6. Prioritized Experience Replay (PER) — `replay_memory.py`
Uniform random sampling treats all stored transitions as equally informative. PER assigns each transition a priority proportional to its absolute TD error (how "surprising" it was to the agent). Transitions that the network has not yet learned well are sampled more frequently, improving sample efficiency. New transitions receive the current maximum priority to guarantee they are sampled at least once. Priorities are updated after every learning step via `update_priorities()`.

### 7. Multi-Head Self-Attention — `models.py`
The custom single-head `SelfAttention` class was replaced with PyTorch's built-in `nn.MultiheadAttention` using 4 heads. Multi-head attention lets the encoder simultaneously attend to different positional and feature subspaces (e.g. conserved regions, gap patterns, base-pairing motifs), producing richer sequence representations that help the DQN converge faster. PyTorch's fused attention kernel also runs faster than the manual implementation. `d_k` and `d_v` are retained in the `Encoder` signature for backward compatibility but are no longer used; head dimension is derived as `d_model // n_heads` (64 // 4 = 16).

### 8. `torch.compile()` — `dqn.py`
Both `eval_net` and `target_net` are passed through `torch.compile()` (available in PyTorch ≥ 2.0). This JIT-compiles the model's computation graph, fuses operations, and eliminates Python interpreter overhead for an additional 10–30% speedup with zero algorithmic changes. A `try/except` block ensures the code falls back gracefully on older PyTorch versions.

## Embedding Improvements

### A. IUPAC Ambiguous Nucleotide Token IDs — `env.py`, `models.py`, `dqn.py`
The original `nucleotides_map` incorrectly collapsed all IUPAC ambiguity codes onto canonical base IDs:

| Code | Meaning | Old mapping | Problem |
|---|---|---|---|
| N | any nucleotide (A, T, C, G) | → A (ID 1) | loses all ambiguity information |
| R | purine (A or G)             | → A (ID 1) | treated identically to plain A |
| W | weak bond (A or T)          | → A (ID 1) | treated identically to plain A |
| K | keto (G or T)               | → T (ID 2) | treated identically to plain T |
| Y | pyrimidine (C or T)         | → T (ID 2) | treated identically to plain T |

Each ambiguous code now has its own unique token ID (N=6, R=7, W=8, K=9, Y=10), expanding the vocabulary from 6 to 11 tokens. A `VOCAB_SIZE = 11` constant is exported from `env.py` and imported in `dqn.py` so that the `Encoder`'s embedding table size stays in sync automatically.

In `Encoder.__init__()`, `_init_iupac_embeddings()` seeds each ambiguous embedding row with the mean of its constituent base embeddings (e.g. R's embedding starts as `(emb[A] + emb[G]) / 2`). This gives the model a biologically meaningful starting point rather than random noise for these tokens, while still allowing the embeddings to be fine-tuned during training.

## Additional Deep RL Algorithms

Two on-policy algorithms were added alongside DQN without removing it. All three share the same `Environment`, `Encoder`, and `PositionalEncoding`. Select an algorithm at runtime with the `--algorithm` flag:

```bash
python main.py sequences.fasta                   # DQN (default, original behaviour)
python main.py sequences.fasta --algorithm a2c   # Advantage Actor-Critic
python main.py sequences.fasta --algorithm ppo   # Proximal Policy Optimisation
```

### File overview

| File | Role |
|---|---|
| `dqn.py` | Original DQN agent (unchanged interface) |
| `actor_critic.py` | `ActorCriticNet` shared network + `ActorCritic` (A2C) agent |
| `ppo.py` | `PPO` agent (imports `ActorCriticNet` from `actor_critic.py`) |
| `config.py` | Added `a2c_*` and `ppo_*` hyperparameter groups |
| `main.py` | Added `--algorithm` flag, `train_a2c()`, `train_ppo()`, per-algorithm parameter printers |

### Shared architecture — `ActorCriticNet`

Both A2C and PPO use the same network defined in `actor_critic.py`: the same Transformer encoder from `models.py` (with multi-head attention and IUPAC-aware embeddings) feeds into two independent heads:

- **Actor head**: two linear layers (dim·d_model → 512 → action_number) that output logits for a `Categorical` distribution over gap-insertion actions.
- **Critic head**: two linear layers (dim·d_model → 512 → 1) that output a scalar state value V(s).

Sharing the encoder backbone means both heads benefit from the same learned sequence features while specialising independently.

### A2C — `actor_critic.py`

**How it differs from DQN:**

| | DQN | A2C |
|---|---|---|
| Policy | ε-greedy (implicit) | Explicit categorical π(a\|s) |
| Exploration | ε-decay | Entropy bonus H[π(·\|s)] |
| Data reuse | Replay buffer (off-policy) | Episode discarded after update (on-policy) |
| Variance reduction | Target network | Critic baseline V(s) |
| Update frequency | Every step | Once per episode |

**Update rule (one gradient step per episode):**

```
R_t      = r_t + γ · done_t · R_{t+1}          (discounted return)
A(s,a)   = R_t − V(s_t)                         (advantage)
L        = −E[log π(a|s) · Â]                   (actor: maximise advantage)
         + value_coef  · Huber(V(s), R)          (critic: fit returns)
         − entropy_coef · H[π(·|s)]             (regularise: stay exploratory)
```

Advantages are normalised per episode (`mean=0, std=1`) for stable gradient magnitudes across episodes of different lengths.

### PPO — `ppo.py`

PPO extends A2C with three improvements:

**1. Generalised Advantage Estimation (GAE, λ=0.95)**
Instead of one-step TD advantages, GAE computes a weighted sum of k-step returns that smoothly interpolates between TD(0) (λ=0, low variance, high bias) and Monte-Carlo (λ=1, high variance, zero bias):
```
δ_t  = r_t + γ · done_t · V(s_{t+1}) − V(s_t)
Â_t  = δ_t + γλ · done_t · Â_{t+1}
```

**2. Clipped surrogate objective**
The ratio r_t = π_new(a|s) / π_old(a|s) measures how much the policy has changed. Clipping to [1−ε, 1+ε] (default ε=0.2) prevents a single update from changing the policy too drastically:
```
L_CLIP = E[min(r_t · Â_t,  clip(r_t, 1−ε, 1+ε) · Â_t)]
```

**3. Multiple update epochs**
After collecting one episode, PPO runs `ppo_epochs=4` gradient steps on the same data. The clip constraint keeps successive updates safe and prevents overfitting to the old-policy batch.

**New `config.py` hyperparameters:**

| Parameter | Default | Description |
|---|---|---|
| `a2c_lr` | 0.0003 | A2C Adam learning rate |
| `a2c_value_coef` | 0.5 | Weight of critic loss |
| `a2c_entropy_coef` | 0.01 | Entropy bonus weight |
| `ppo_lr` | 0.0003 | PPO Adam learning rate |
| `ppo_clip_eps` | 0.2 | Clip range ε for PPO |
| `ppo_epochs` | 4 | Update epochs per rollout |
| `ppo_value_coef` | 0.5 | Weight of critic loss |
| `ppo_entropy_coef` | 0.01 | Entropy bonus weight |
| `gae_lambda` | 0.95 | GAE λ |

All three algorithms support `torch.compile`, mixed-precision (`torch.cuda.amp`), and gradient clipping.

## Parallel Environments

### Motivation

The training bottleneck for on-policy algorithms (A2C, PPO) is not the gradient update — it is **data collection**. With a single environment, every forward pass processes exactly one state vector. Modern GPUs can process a batch of N state vectors in nearly the same wall-clock time as one, so running N environments simultaneously and stacking their states into a single `(N, state_dim)` tensor gives roughly N× data throughput at negligible extra cost.

For DQN the benefit is different: N envs fill the replay buffer N× faster, so the agent reaches the minimum batch size sooner and begins learning from more diverse experience earlier in training.

### New file: `parallel_env.py`

`ParallelEnvironment` wraps `n_envs` independent `Environment` instances:

```
par_env = ParallelEnvironment(sequences, n_envs=4)
states  = par_env.reset_all()          # list of 4 states

while not par_env.all_done():
    active_idx = [i for i,a in enumerate(par_env.active_mask) if a]
    # ONE forward pass on a (|active|, state_dim) tensor
    actions = batched_forward(active_states)
    results = par_env.step_all(actions)
```

When an env finishes its episode (`done==0`) it is marked inactive; `step_all()` returns `(0.0, None, 0)` for inactive slots so callers can safely iterate the full list.

### Changes to agents

| File | Addition |
|---|---|
| `dqn.py` | `select_batch(states)` — batched epsilon-greedy; random actions are sampled independently, greedy actions use one forward pass |
| `actor_critic.py` | `update_parallel(traj_s, traj_a, traj_r, traj_v, traj_d)` — computes per-trajectory returns, concatenates, normalises across all N trajectories, one gradient step |
| `actor_critic.py` | `_gradient_step()` — extracted shared update logic (used by both `update()` and `update_parallel()`) |
| `ppo.py` | `update_parallel(traj_s, traj_a, traj_lp, traj_r, traj_v, traj_d)` — per-trajectory GAE, concatenate, ppo_epochs clipped updates |
| `ppo.py` | `_gae_for_traj()` (static), `_gradient_steps()` — extracted from `update()` for reuse |

### New training functions in `main.py`

| Function | Algorithm | What changes |
|---|---|---|
| `train_dqn_parallel()` | DQN | `select_batch()` per step; all N transitions pushed to shared replay buffer |
| `train_a2c_parallel()` | A2C | `_collect_episode_parallel()` → `update_parallel()` |
| `train_ppo_parallel()` | PPO | `_collect_episode_parallel()` (with log_probs) → `update_parallel()` |

`_collect_episode_parallel()` is a shared helper in `main.py` that runs the episode loop across all N envs using a caller-supplied `forward_fn`, accumulating per-env trajectory dicts.

### Dispatch

Parallel training is automatic when `n_envs > 1` in `config.py` (default: 4):

```bash
python main.py sequences.fasta --algorithm ppo   # uses train_ppo_parallel (n_envs=4)
```

Set `n_envs = 1` in `config.py` to revert to the original single-env loops.

## Profile-Based Scoring

### Motivation

The original `calc_score()` method scores the final alignment using Sum-of-Pairs (SP): for every column it iterates over all C(k, 2) pairs of sequences, giving O(L·k²) time. For alignments with many sequences, this inner loop becomes measurably slow. A PSSM-based approach achieves an equivalent score in O(L·k) time by exploiting C(n, 2) combinatorics — replacing the pairwise loop with a single per-column token count.

### New methods — `env.py`

| Method | Complexity | Description |
|---|---|---|
| `build_pssm()` | O(L·k) | Returns a `(L, VOCAB_SIZE)` numpy array of per-position nucleotide frequencies |
| `calc_profile_score()` | O(L·k) | SP-equivalent score computed via C(n, 2) combinatorics on per-token counts |

**`calc_profile_score()` derivation** — for each alignment column with k sequences:
```
g  = count of gap tokens (token ID 5)
n  = k − g  (non-gap count)

gap_pairs      = g*(k−g) + C(g,2)     # gap vs non-gap + gap vs gap
match_pairs    = Σ C(count[nuc], 2)   # for each distinct non-gap token
mismatch_pairs = C(n, 2) − match_pairs

score_i = GAP_PENALTY*gap_pairs + MATCH_REWARD*match_pairs + MISMATCH_PENALTY*mismatch_pairs
```

Because `gap_pairs + match_pairs + mismatch_pairs = C(k, 2)`, this is *mathematically identical* to iterating all pairs — just O(k) per column instead of O(k²).

### `--scoring` flag — `main.py`

Select the scoring function at runtime:

```bash
python main.py sequences.fasta                          # sp (default)
python main.py sequences.fasta --scoring profile        # profile score + PSSM table
python main.py sequences.fasta --scoring both           # both scores + PSSM table
python main.py sequences.fasta --algorithm ppo --scoring both
```

| Flag value | Output |
|---|---|
| `sp` (default) | Sum-of-Pairs score only |
| `profile` | Profile score + PSSM frequency table |
| `both` | SP score, profile score, and PSSM table |

The `--scoring` flag works with all three algorithms (DQN, A2C, PPO) and both single-env and parallel training modes. The new `_print_results(env, scoring)` helper in `main.py` centralises all result reporting, replacing the previously duplicated output blocks across all six training functions.

### B. Embedding Scale (`√d_model`) — `models.py`
Following Vaswani et al. (2017) *"Attention Is All You Need"*, token embeddings are now multiplied by `√d_model` (= 8 for `d_model=64`) before positional encoding is added. Without this scaling, the positional encoding signal — whose magnitude is fixed near 1 by the sinusoidal formula — overwhelms the token embedding signal, which tends to be small from random initialisation. Scaling the embeddings up restores the intended balance between token identity and positional information.
