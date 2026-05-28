#!/usr/bin/env python3
"""
grid_search.py  —  RL Algorithm × Reward-Signal Grid Search
=============================================================
Trains and evaluates all 6 combinations of:
  algorithms  : DQN, A2C, PPO
  reward types: sp (default pairwise SP), profile (entropy-conservation)

on a small sample of test cases, then generates comparison plots.

Reward signals
--------------
  sp (default)
    Per-step reward = sum-of-pairs score for the last aligned column.
    Penalises every mismatching pair equally.  This is the current built-in
    __calc_reward() in Environment.

  profile (conservation)
    Per-step reward based on the *entropy* of the column's nucleotide
    distribution, scaled to the same numeric range as SP.

    For the last column with k sequences (g gaps, n = k-g non-gap):
      conservation = 1 - H_norm
        where H_norm = H(non-gap freqs) / log(min(k, 4))   ∈ [0, 1]
      conservation_reward = conservation * C(k,2) * MATCH_REWARD
      gap_penalty         = g * GAP_PENALTY
      reward              = conservation_reward + gap_penalty

    Key difference from SP: SP penalises every mismatch pair equally,
    while profile gives a *proportional* reward — columns with one
    dominant nucleotide score high even if a minority of sequences
    disagree, whereas SP docks a fixed penalty per mismatching pair.

Usage
-----
  python grid_search.py [--dataset 3x30]          # which dataset to sample from
                        [--n_tests  10]            # number of test cases
                        [--episodes 80]            # training episodes per run
                        [--outdir   figures/grid]  # figure output directory
                        [--csv      results/grid_search.csv]
"""

import os
import sys

# ── OpenMP / threading guards (must come before any torch import) ─────────────
# Multiple copies of the OpenMP runtime are linked by different conda packages.
# Setting KMP_DUPLICATE_LIB_OK suppresses the fatal error; limiting PyTorch to
# 1 intra-op thread per process avoids the pthread_mutex_init crash that follows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import math
import time
import argparse
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from Bio import SeqIO
import torch
torch.set_num_threads(1)

warnings.filterwarnings("ignore")

# ── disable torch.compile globally ───────────────────────────────────────────
# torch.compile() links against the OpenMP runtime a second time on macOS,
# triggering a pthread_mutex_init failure when multiple DQN instances are
# created in the same process.  Replacing it with an identity function is safe
# here because grid_search runs on CPU and gains nothing from JIT compilation.
torch.compile = lambda model, *a, **kw: model  # type: ignore[assignment]

# ── project imports ───────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config
from env import Environment
from dqn import DQN
from actor_critic import ActorCritic
from ppo import PPO
from acer import ACER

# ── constants ─────────────────────────────────────────────────────────────────
ALGORITHMS    = ["DQN", "A2C", "PPO", "ACER"]
REWARD_TYPES  = ["sp", "profile"]
GAP_TOKEN     = 5

ALGO_COLORS = {
    "DQN":  "#2196F3",   # blue
    "A2C":  "#4CAF50",   # green
    "PPO":  "#FF9800",   # orange
    "ACER": "#9C27B0",   # purple
}
REWARD_HATCHES = {
    "sp":      "",
    "profile": "//",
}
REWARD_LABELS = {
    "sp":      "SP (default)",
    "profile": "Profile (entropy-conservation)",
}

DATASET_FASTA_DIRS = {
    "3x30":  os.path.join(ROOT, "datasets", "fasta_files", "dataset1_3x30bp"),
    "6x30":  os.path.join(ROOT, "datasets", "fasta_files", "dataset1_6x30bp"),
    "6x60":  os.path.join(ROOT, "datasets", "fasta_files", "dataset1_6x60bp"),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  PROFILE REWARD  (entropy-conservation)
# ═══════════════════════════════════════════════════════════════════════════════

def _profile_step_reward(env: Environment) -> float:
    """
    Entropy-based conservation reward for the most recently aligned column.

    Differs from SP (pairwise sum) in how it handles partially-conserved columns:
      - SP: fixed MISMATCH_PENALTY per pair → docks equally for each bad pair
      - Profile: proportional to conservation → a column 2/3 conserved still
                 scores ~67 % of a perfect column's reward rather than losing
                 C(mismatches,2) × MISMATCH_PENALTY

    Scales to the same numeric range as SP for perfect / all-gap columns,
    so training hyperparameters (learning rate, max_reward sentinel, etc.) need
    no tuning.
    """
    col_idx = len(env.aligned[0]) - 1          # index of the newly placed column
    k       = env.row
    c2      = lambda n: n * (n - 1) // 2

    # Collect this column's tokens
    col     = [env.aligned[j][col_idx] for j in range(k)]
    counts  = Counter(col)
    g       = counts.get(GAP_TOKEN, 0)          # gap count
    n       = k - g                             # non-gap count

    # ── gap penalty (same as SP) ──────────────────────────────────────────────
    gap_score = g * config.GAP_PENALTY          # ← per gap, not per gap-pair,
    # to keep the scale identical when g=k (all-gap column)

    if n == 0:
        return float(gap_score)

    # ── conservation component ────────────────────────────────────────────────
    non_gap_counts = [cnt for tok, cnt in counts.items() if tok != GAP_TOKEN]
    total_non_gap  = sum(non_gap_counts)
    probs          = [c / total_non_gap for c in non_gap_counts]

    # Shannon entropy (natural log), normalised by maximum possible entropy
    entropy      = -sum(p * math.log(p + 1e-12) for p in probs)
    max_entropy  = math.log(min(k, 4) + 1e-12)   # uniform over ≤4 nucleotides
    conservation = 1.0 - (entropy / (max_entropy + 1e-12))  # ∈ [0, 1]

    # Scale: perfect conservation → same reward as C(k,2) SP matches
    conservation_score = conservation * c2(k) * config.MATCH_REWARD

    return float(conservation_score + gap_score)


def _patch_profile_reward(env: Environment) -> None:
    """
    Monkey-patch env so that step() uses the profile reward instead of SP.
    Works by replacing the name-mangled _Environment__calc_reward attribute.
    """
    env._Environment__calc_reward = lambda: _profile_step_reward(env)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_test_cases(dataset_key: str, n: int) -> list:
    """
    Return a list of up to n dicts: {name: str, seqs: list[str]}
    Reads individual test*.fasta files from the dataset's fasta_files directory.
    """
    fasta_dir = DATASET_FASTA_DIRS[dataset_key]
    if not os.path.isdir(fasta_dir):
        raise FileNotFoundError(f"FASTA directory not found: {fasta_dir}")

    # Sort numerically: test0, test1, …, test9, test10, …
    files = sorted(
        [f for f in os.listdir(fasta_dir) if f.endswith(".fasta")],
        key=lambda f: int("".join(filter(str.isdigit, f)) or "0")
    )[:n]

    cases = []
    for fname in files:
        path  = os.path.join(fasta_dir, fname)
        seqs  = [str(r.seq) for r in SeqIO.parse(path, "fasta")]
        name  = os.path.splitext(fname)[0]   # "test0"
        cases.append({"name": name, "seqs": seqs})

    return cases


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING FUNCTIONS
#  Each returns the final SP score (int) after training + greedy inference.
# ═══════════════════════════════════════════════════════════════════════════════

def _run_inference(agent, env: Environment) -> int:
    """Greedy rollout on a fresh env.reset(); returns SP score after padding."""
    state = env.reset()
    while True:
        action = agent.predict(state)
        _, next_state, done = env.step(action)
        state = next_state
        if done == 0:
            break
    env.padding()
    return env.calc_score()


def run_dqn(seqs: list, reward_type: str, episodes: int, d_model: int = 64) -> int:
    env   = Environment(seqs)
    if reward_type == "profile":
        _patch_profile_reward(env)

    # Pass d_model through to the underlying Net so grid search can use a
    # smaller model (e.g. d_model=16) for fast CPU runs.
    import dqn as _dqn_mod
    orig_init = _dqn_mod.Net.__init__
    def _patched_init(self, sn, msl, an, mv, dm=d_model):
        orig_init(self, sn, msl, an, mv, dm)
    _dqn_mod.Net.__init__ = _patched_init

    agent = DQN(env.action_number, env.row, env.max_len,
                env.max_len * env.max_reward)
    _dqn_mod.Net.__init__ = orig_init  # restore immediately after construction

    for _ in range(episodes):
        state = env.reset()
        while True:
            action        = agent.select(state)
            rew, ns, done = env.step(action)
            agent.replay_memory.push((state, ns, action, rew, done))
            agent.update()
            if done == 0:
                break
            state = ns
        agent.update_epsilon()

    return _run_inference(agent, env)


def run_a2c(seqs: list, reward_type: str, episodes: int, d_model: int = 64) -> int:
    env   = Environment(seqs)
    if reward_type == "profile":
        _patch_profile_reward(env)

    import actor_critic as _ac_mod
    orig_init = _ac_mod.ActorCriticNet.__init__
    def _patched_init(self, sn, msl, an, dm=d_model):
        orig_init(self, sn, msl, an, dm)
    _ac_mod.ActorCriticNet.__init__ = _patched_init

    agent = ActorCritic(env.action_number, env.row, env.max_len)
    _ac_mod.ActorCriticNet.__init__ = orig_init

    for _ in range(episodes):
        state = env.reset()
        while True:
            action        = agent.select(state)
            rew, ns, done = env.step(action)
            agent.record_transition(rew, float(done))
            if done == 0:
                break
            state = ns
        agent.update()

    return _run_inference(agent, env)


def run_ppo(seqs: list, reward_type: str, episodes: int, d_model: int = 64) -> int:
    env   = Environment(seqs)
    if reward_type == "profile":
        _patch_profile_reward(env)

    import actor_critic as _ac_mod
    orig_init = _ac_mod.ActorCriticNet.__init__
    def _patched_init(self, sn, msl, an, dm=d_model):
        orig_init(self, sn, msl, an, dm)
    _ac_mod.ActorCriticNet.__init__ = _patched_init

    agent = PPO(env.action_number, env.row, env.max_len)
    _ac_mod.ActorCriticNet.__init__ = orig_init

    for _ in range(episodes):
        state = env.reset()
        while True:
            action        = agent.select(state)
            rew, ns, done = env.step(action)
            agent.record_transition(rew, float(done))
            if done == 0:
                break
            state = ns
        agent.update()

    return _run_inference(agent, env)


def run_acer(seqs: list, reward_type: str, episodes: int, d_model: int = 64) -> int:
    """Train ACER and return the final SP score after greedy inference.

    ACER directly accepts d_model — no monkey-patching needed.
    Each call to agent.update() performs:
      • 1 on-policy gradient step on the just-completed episode
      • acer_replay_ratio off-policy steps from random buffered episodes
      • EMA update of the average policy network
    """
    env = Environment(seqs)
    if reward_type == "profile":
        _patch_profile_reward(env)

    agent = ACER(env.action_number, env.row, env.max_len, d_model)

    for _ in range(episodes):
        state = env.reset()
        while True:
            action        = agent.select(state)
            rew, ns, done = env.step(action)
            agent.record_transition(rew, float(done))
            if done == 0:
                break
            state = ns
        agent.update()

    return _run_inference(agent, env)


_RUNNERS = {"DQN": run_dqn, "A2C": run_a2c, "PPO": run_ppo, "ACER": run_acer}


# ═══════════════════════════════════════════════════════════════════════════════
#  GRID SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def run_grid(test_cases: list, episodes: int, d_model: int = 64) -> pd.DataFrame:
    """
    Run all 6 (algo × reward) combinations on every test case.
    Returns a long-format DataFrame: test_id | algo | reward | sp_score | time_s
    """
    n_combos = len(ALGORITHMS) * len(REWARD_TYPES)
    n_total  = n_combos * len(test_cases)
    rows     = []

    config.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.device      = torch.device(config.device_name)
    # Force single-env mode — grid search runs training sequentially
    config.n_envs = 1
    # Scale DQN buffer/batch down so the replay buffer fills quickly and
    # learning starts immediately.  Original values (batch=256, mem=10000)
    # need hundreds of episodes to see any gradient steps on CPU.
    config.batch_size          = 32
    config.replay_memory_size  = 200
    config.update_iteration    = 4

    print(f"\n  device  : {config.device_name}")
    print(f"  d_model : {d_model}  (transformer hidden size)")
    print(f"  episodes: {episodes}")
    print(f"  combos  : {n_combos}  ({' × '.join(ALGORITHMS)} × {' × '.join(REWARD_TYPES)})")
    print(f"  tests   : {len(test_cases)}")
    print(f"  total   : {n_total} training runs\n")

    bar = tqdm(total=n_total, ncols=80, unit="run")

    for algo in ALGORITHMS:
        runner = _RUNNERS[algo]
        for reward_type in REWARD_TYPES:
            for tc in test_cases:
                bar.set_description(f"{algo}+{reward_type} {tc['name']}")
                t0      = time.monotonic()
                sp      = runner(tc["seqs"], reward_type, episodes, d_model)
                elapsed = time.monotonic() - t0
                rows.append({
                    "test_id":     tc["name"],
                    "algo":        algo,
                    "reward":      reward_type,
                    "combo":       f"{algo}+{reward_type}",
                    "sp_score":    sp,
                    "time_s":      round(elapsed, 2),
                })
                bar.update(1)

    bar.close()
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig, outdir, fname):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"    saved → {path}")
    plt.close(fig)


def _combo_color(algo: str, reward: str) -> tuple:
    base   = mcolors.to_rgb(ALGO_COLORS[algo])
    # profile variant: slightly lighter
    factor = 1.0 if reward == "sp" else 0.65
    return tuple(min(1.0, c + (1 - c) * (1 - factor)) for c in base)


def fig_grouped_bar(df: pd.DataFrame, outdir: str):
    """Mean SP ± SEM per (algo, reward) pair — grouped by algo."""
    print("  fig1: grouped bar …")
    combos  = [(a, r) for a in ALGORITHMS for r in REWARD_TYPES]
    means   = [df[(df.algo == a) & (df.reward == r)]["sp_score"].mean() for a, r in combos]
    sems    = [df[(df.algo == a) & (df.reward == r)]["sp_score"].sem()  for a, r in combos]
    colors  = [_combo_color(a, r) for a, r in combos]
    labels  = [f"{a}\n{REWARD_LABELS[r]}" for a, r in combos]

    fig, ax = plt.subplots(figsize=(11, 5))
    x       = np.arange(len(combos))
    bars    = ax.bar(x, means, yerr=sems, color=colors, alpha=0.9,
                     capsize=5, error_kw={"linewidth": 1.3, "ecolor": "#333"},
                     edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean SP Score  (higher = better)")
    ax.set_title("Grid Search: Mean SP Score per Algorithm × Reward Type\n"
                 "(error bars = ±1 SEM across test cases)", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", alpha=0.3)

    # annotate
    for bar, mean, sem in zip(bars, means, sems):
        if not np.isnan(mean):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    mean - abs(mean) * 0.03,
                    f"{mean:.0f}", ha="center", va="top", fontsize=8.5,
                    fontweight="bold")

    # legend: reward type
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#AAB", label="SP reward (default)"),
        Patch(facecolor="#EE9", label="Profile reward (entropy-conservation)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")

    fig.tight_layout()
    _save(fig, outdir, "grid1_grouped_bar.png")


def fig_box(df: pd.DataFrame, outdir: str):
    """Box + strip plot — distribution of SP across test cases."""
    print("  fig2: box plots …")
    combos = [(a, r) for a in ALGORITHMS for r in REWARD_TYPES]
    data   = [df[(df.algo == a) & (df.reward == r)]["sp_score"].values for a, r in combos]
    colors = [_combo_color(a, r) for a, r in combos]
    labels = [f"{a}\n({r})" for a, r in combos]

    fig, ax = plt.subplots(figsize=(11, 5))

    bp = ax.boxplot(data, positions=range(len(combos)), patch_artist=True,
                    medianprops={"color": "#222", "linewidth": 1.8},
                    whiskerprops={"linewidth": 1.2},
                    boxprops={"linewidth": 1.2},
                    flierprops={"marker": "x", "markersize": 5, "alpha": 0.5})
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    # jitter overlay
    for i, vals in enumerate(data):
        jitter = np.random.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=colors[i], alpha=0.6, s=22, zorder=4, edgecolors="white",
                   linewidths=0.5)

    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("SP Score")
    ax.set_title("SP Score Distribution per Algorithm × Reward Type\n"
                 "(dots = individual test cases)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, outdir, "grid2_box_distribution.png")


def fig_heatmap(df: pd.DataFrame, outdir: str):
    """Mean SP heatmap: rows = algorithms, cols = reward types."""
    print("  fig3: heatmap …")
    pivot = df.groupby(["algo", "reward"])["sp_score"].mean().unstack()
    # order axes
    pivot = pivot.reindex(index=ALGORITHMS,
                          columns=[r for r in REWARD_TYPES if r in pivot.columns])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Mean SP Score — Algo × Reward Heatmaps", fontsize=13, fontweight="bold")

    # absolute SP
    ax = axes[0]
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([REWARD_LABELS[r] for r in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_title("Mean SP Score (absolute)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if abs(v) > 0.6 * abs(pivot.values).max() else "black")

    # relative: % improvement of profile over sp per algo
    ax = axes[1]
    if "sp" in pivot.columns and "profile" in pivot.columns:
        rel = ((pivot["profile"] - pivot["sp"]) / pivot["sp"].abs() * 100).values.reshape(-1, 1)
        im2 = ax.imshow(rel, cmap="RdYlGn", aspect="auto",
                        vmin=-max(abs(rel).max(), 1), vmax=max(abs(rel).max(), 1))
        ax.set_xticks([0])
        ax.set_xticklabels(["Profile vs SP\n(% change)"], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10)
        ax.set_title("Profile reward vs SP reward\n(% SP score change)")
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label="% change")
        for i, algo in enumerate(pivot.index):
            v = rel[i, 0]
            if not np.isnan(v):
                ax.text(0, i, f"{v:+.1f}%", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if abs(v) > 5 else "black")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Both reward types needed\nfor comparison",
                ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    _save(fig, outdir, "grid3_heatmap.png")


def fig_per_test(df: pd.DataFrame, outdir: str):
    """Line plot of SP score per test case for each combination."""
    print("  fig4: per-test lines …")
    combos = [(a, r) for a in ALGORITHMS for r in REWARD_TYPES]

    def test_num(tid):
        import re
        m = re.search(r"(\d+)$", str(tid))
        return int(m.group(1)) if m else 0

    fig, axes = plt.subplots(len(ALGORITHMS), 1,
                              figsize=(12, 4 * len(ALGORITHMS)), sharex=True)
    fig.suptitle("SP Score per Test Case — All Combinations", fontsize=13, fontweight="bold")

    for ax, algo in zip(axes, ALGORITHMS):
        for reward in REWARD_TYPES:
            sub = (df[(df.algo == algo) & (df.reward == reward)]
                   .copy()
                   .assign(test_num=lambda d: d["test_id"].apply(test_num))
                   .sort_values("test_num"))
            ls  = "-" if reward == "sp" else "--"
            col = _combo_color(algo, reward)
            ax.plot(sub["test_num"], sub["sp_score"],
                    label=REWARD_LABELS[reward],
                    color=ALGO_COLORS[algo], linestyle=ls, linewidth=2,
                    alpha=0.85, marker="o", markersize=5,
                    markerfacecolor=col, markeredgecolor="white",
                    markeredgewidth=0.8)
        ax.set_title(f"{algo}", fontsize=11, color=ALGO_COLORS[algo],
                     fontweight="bold")
        ax.set_ylabel("SP Score")
        ax.legend(fontsize=8.5, loc="upper right")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Test index")
    fig.tight_layout()
    _save(fig, outdir, "grid4_per_test_lines.png")


def fig_reward_delta(df: pd.DataFrame, outdir: str):
    """
    Paired scatter: for each (algo, test_case), plot SP vs Profile reward SP.
    Points above the y=x line → profile reward did better.
    """
    print("  fig5: reward delta scatter …")
    if "sp" not in df.reward.values or "profile" not in df.reward.values:
        print("    [skip] both reward types needed.")
        return

    fig, axes = plt.subplots(1, len(ALGORITHMS), figsize=(5 * len(ALGORITHMS), 5))
    fig.suptitle("Profile reward vs SP reward — Paired per Test Case\n"
                 "Points above y = x: profile reward produced a better alignment",
                 fontsize=12, fontweight="bold")

    for ax, algo in zip(axes, ALGORITHMS):
        sp_vals   = (df[(df.algo == algo) & (df.reward == "sp")]
                     .set_index("test_id")["sp_score"])
        prof_vals = (df[(df.algo == algo) & (df.reward == "profile")]
                     .set_index("test_id")["sp_score"])
        common    = sp_vals.index.intersection(prof_vals.index)

        x, y = sp_vals[common].values, prof_vals[common].values
        ax.scatter(x, y, color=ALGO_COLORS[algo], alpha=0.75, s=60,
                   edgecolors="white", linewidths=0.6)

        lo = min(x.min(), y.min()) * 1.05
        hi = max(x.max(), y.max()) * 0.95
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="y = x")

        n_profile_better = (y > x).sum()
        ax.text(0.05, 0.95,
                f"Profile better: {n_profile_better}/{len(common)}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        ax.set_xlabel("SP reward → final SP score")
        ax.set_ylabel("Profile reward → final SP score")
        ax.set_title(algo, fontsize=11, color=ALGO_COLORS[algo], fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, outdir, "grid5_reward_delta_scatter.png")


def fig_timing(df: pd.DataFrame, outdir: str):
    """Bar chart of mean training time per combination."""
    print("  fig6: timing …")
    combos  = [(a, r) for a in ALGORITHMS for r in REWARD_TYPES]
    means   = [df[(df.algo == a) & (df.reward == r)]["time_s"].mean() for a, r in combos]
    colors  = [_combo_color(a, r) for a, r in combos]
    labels  = [f"{a}\n({r})" for a, r in combos]

    fig, ax = plt.subplots(figsize=(9, 4))
    x       = np.arange(len(combos))
    bars    = ax.bar(x, means, color=colors, alpha=0.88,
                     edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean training time (s) per test case")
    ax.set_title("Training Time per Algorithm × Reward Type", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    for bar, mean in zip(bars, means):
        if not np.isnan(mean):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    mean + 0.2, f"{mean:.1f}s",
                    ha="center", fontsize=8.5)
    fig.tight_layout()
    _save(fig, outdir, "grid6_timing.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DPAMSA Grid Search: RL algorithm × reward type",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset",  default="3x30",
                        choices=list(DATASET_FASTA_DIRS),
                        help="Dataset to sample test cases from (default: 3x30)")
    parser.add_argument("--n_tests",  type=int, default=10,
                        help="Number of test cases to use (default: 10)")
    parser.add_argument("--episodes", type=int, default=60,
                        help="Training episodes per run (default: 60)")
    parser.add_argument("--outdir",   default=os.path.join(ROOT, "figures", "grid"),
                        help="Output directory for figures")
    parser.add_argument("--csv",      default=os.path.join(ROOT, "results", "grid_search.csv"),
                        help="Path for the results CSV")
    parser.add_argument("--d_model",  type=int, default=16,
                        help="Transformer hidden size (default: 16 for fast CPU runs; "
                             "use 64 to match full model)")
    args = parser.parse_args()

    print("=" * 60)
    print("  DPAMSA Grid Search")
    print("=" * 60)

    # ── load test data ────────────────────────────────────────────────────────
    print(f"\n[1/3]  Loading {args.n_tests} test cases from dataset1_{args.dataset}bp …")
    test_cases = load_test_cases(args.dataset, args.n_tests)
    print(f"       loaded {len(test_cases)} cases  "
          f"({len(test_cases[0]['seqs'])} seqs × "
          f"{len(test_cases[0]['seqs'][0])} bp each)")

    # ── run grid search ───────────────────────────────────────────────────────
    print(f"\n[2/3]  Running grid search ({args.episodes} episodes/run) …")
    # Temporarily override max_episode so the training loops respect --episodes
    orig_max = config.max_episode
    config.max_episode = args.episodes
    try:
        df = run_grid(test_cases, args.episodes, args.d_model)
    finally:
        config.max_episode = orig_max

    # ── save results CSV ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    df.to_csv(args.csv, index=False)
    print(f"\n       results saved → {args.csv}")

    # ── summary table ─────────────────────────────────────────────────────────
    summary = (df.groupby(["algo", "reward"])["sp_score"]
                 .agg(mean="mean", std="std", median="median",
                      min="min", max="max", count="count")
                 .reset_index()
                 .round(1))
    print("\n" + summary.to_string(index=False))

    # best combo
    best_row = summary.loc[summary["mean"].idxmax()]
    print(f"\n  ★  Best combo: {best_row['algo']} + {best_row['reward']} reward  "
          f"(mean SP = {best_row['mean']:.1f})")

    # ── generate figures ──────────────────────────────────────────────────────
    print(f"\n[3/3]  Generating figures → {args.outdir}")
    fig_grouped_bar(df, args.outdir)
    fig_box(df, args.outdir)
    fig_heatmap(df, args.outdir)
    fig_per_test(df, args.outdir)
    fig_reward_delta(df, args.outdir)
    fig_timing(df, args.outdir)

    print(f"\nDone!  Figures → {args.outdir}")
    print(f"       Results → {args.csv}")


if __name__ == "__main__":
    main()
