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

### B. Embedding Scale (`√d_model`) — `models.py`
Following Vaswani et al. (2017) *"Attention Is All You Need"*, token embeddings are now multiplied by `√d_model` (= 8 for `d_model=64`) before positional encoding is added. Without this scaling, the positional encoding signal — whose magnitude is fixed near 1 by the sinusoidal formula — overwhelms the token embedding signal, which tends to be small from random initialisation. Scaling the embeddings up restores the intended balance between token identity and positional information.
