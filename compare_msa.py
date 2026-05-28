#!/usr/bin/env python3
"""
compare_msa.py  —  Unified MSA Algorithm Comparison Pipeline
=============================================================
Parses all benchmark CSVs and RPT reports under result/,
computes profile scores from DPAMSA/GA-DPAMSA alignments,
builds a master comparison table, and generates publication-
quality figures comparing all available methods.

Supported methods
-----------------
  DPAMSA          – RL agent (from benchmark CSV)
  GA-DPAMSA       – Genetic-algorithm agent (from benchmark CSV + GA RPT)
  ClustalOmega    – from clustalo_msaprobs_clustalw CSV
  MSAProbs        – from clustalo_msaprobs_clustalw CSV
  ClustalW        – from clustalo_msaprobs_clustalw CSV
  DPAMSA Profile  – profile-based recomputation of DPAMSA alignments
                    (validation only: should equal DPAMSA SP exactly)

Datasets
--------
  dataset1_3x30bp – 3 sequences × 30 bp, 50 test cases
  dataset1_6x30bp – 6 sequences × 30 bp, 50 test cases
  dataset1_6x60bp – 6 sequences × 60 bp, 50 test cases

Usage
-----
  python compare_msa.py [--outdir figures] [--csv results/comparison_summary.csv]
"""

import os
import re
import argparse
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore", category=FutureWarning)

# ── scoring constants (must match config.py) ──────────────────────────────────
GAP_PENALTY      = -4
MISMATCH_PENALTY = -4
MATCH_REWARD     =  4

# nucleotide → token ID (same vocab as env.py)
_NUC_TO_ID = {
    'A': 1, 'a': 1,
    'T': 2, 't': 2,
    'C': 3, 'c': 3,
    'G': 4, 'g': 4,
    '-': 5,
    'N': 6, 'n': 6,
    'R': 7, 'r': 7,
    'W': 8, 'w': 8,
    'K': 9, 'k': 9,
    'Y': 10, 'y': 10,
}
GAP_TOKEN = 5

# ── colour palette ────────────────────────────────────────────────────────────
METHOD_COLORS = {
    "DPAMSA":         "#2196F3",   # blue
    "GA-DPAMSA":      "#4CAF50",   # green
    "ClustalW":       "#FF5722",   # deep-orange
    "ClustalOmega":   "#FF9800",   # orange
    "MSAProbs":       "#9C27B0",   # purple
    "DPAMSA Profile": "#00BCD4",   # cyan
}

DATASET_LABELS = {
    "dataset1_3x30bp": "3 seq × 30 bp",
    "dataset1_6x30bp": "6 seq × 30 bp",
    "dataset1_6x60bp": "6 seq × 60 bp",
}

# preferred display order (most-interesting first)
_DISPLAY_ORDER = [
    "DPAMSA", "DPAMSA Profile", "GA-DPAMSA",
    "MSAProbs", "ClustalOmega", "ClustalW",
]

ROOT      = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(ROOT, "result", "benchmark")
RPT_DIR_A = os.path.join(ROOT, "result", "reportDPAMSA")
RPT_DIR_B = os.path.join(ROOT, "result", "reportDPAMSA_GA")


# ═══════════════════════════════════════════════════════════════════════════════
#  STANDALONE SCORING  (no env.py import needed)
# ═══════════════════════════════════════════════════════════════════════════════

def _encode(seq: str) -> list:
    """Convert an alignment string → list of integer token IDs."""
    return [_NUC_TO_ID.get(c, 0) for c in seq.strip()]


def _pad_to(encoded: list, L: int) -> list:
    """Right-pad a token list to length L with GAP_TOKEN."""
    return encoded + [GAP_TOKEN] * (L - len(encoded))


def compute_sp_score(seqs: list) -> int:
    """Sum-of-Pairs SP score from alignment strings.  O(L × k²)."""
    if not seqs:
        return 0
    encoded = [_encode(s) for s in seqs]
    L = max(len(s) for s in encoded)
    encoded = [_pad_to(s, L) for s in encoded]
    k = len(encoded)
    total = 0
    for col in range(L):
        col_nucs = [encoded[j][col] for j in range(k)]
        for a, b in combinations(col_nucs, 2):
            if a == GAP_TOKEN or b == GAP_TOKEN:
                total += GAP_PENALTY
            elif a == b:
                total += MATCH_REWARD
            else:
                total += MISMATCH_PENALTY
    return total


def compute_profile_score(seqs: list) -> int:
    """Profile-based SP score via C(n,2) combinatorics.  O(L × k).
    Mathematically equivalent to compute_sp_score()."""
    if not seqs:
        return 0
    encoded = [_encode(s) for s in seqs]
    L = max(len(s) for s in encoded)
    encoded = [_pad_to(s, L) for s in encoded]
    k = len(encoded)

    def c2(n: int) -> int:
        return n * (n - 1) // 2

    total = 0
    for col in range(L):
        counts: dict = {}
        for j in range(k):
            t = encoded[j][col]
            counts[t] = counts.get(t, 0) + 1
        g              = counts.get(GAP_TOKEN, 0)
        n              = k - g
        gap_pairs      = g * n + c2(g)
        match_pairs    = sum(c2(cnt) for tok, cnt in counts.items() if tok != GAP_TOKEN)
        mismatch_pairs = c2(n) - match_pairs
        total += (GAP_PENALTY      * gap_pairs
                + MATCH_REWARD     * match_pairs
                + MISMATCH_PENALTY * mismatch_pairs)
    return total


def compute_cs(seqs: list) -> float:
    """Column Score: fraction of columns where all sequences share the same
    non-gap nucleotide."""
    if not seqs:
        return 0.0
    encoded = [_encode(s) for s in seqs]
    L = max(len(s) for s in encoded)
    encoded = [_pad_to(s, L) for s in encoded]
    perfect = sum(
        1 for col in range(L)
        if len({encoded[j][col] for j in range(len(encoded))}) == 1
        and encoded[0][col] != GAP_TOKEN
        and encoded[0][col] != 0  # not pad
    )
    return perfect / L if L > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  RPT FORMAT A PARSER  (result/reportDPAMSA/)
#
#  Structure (each entry separated by a blank line):
#      NO: testN
#      AL: 32
#      SP: -272
#      EM: 0
#      CS: 0.0
#      QTY: 3
#      #
#      <seq1>
#      <seq2>
#      ...
# ═══════════════════════════════════════════════════════════════════════════════

def parse_dpamsa_rpt(path: str) -> list:
    """Return list of dicts: test_id, sp, em, cs, qty, seqs, profile_sp, computed_cs."""
    with open(path) as fh:
        content = fh.read()

    records = []
    # split into per-entry blocks on blank lines
    blocks = [b.strip() for b in re.split(r'\n\s*\n', content) if b.strip()]

    for block in blocks:
        lines = block.splitlines()
        hdr = {}
        seqs = []
        after_hash = False

        for line in lines:
            stripped = line.strip()
            if stripped == '#':
                after_hash = True
                continue
            if after_hash:
                if stripped:
                    seqs.append(stripped)
            else:
                m = re.match(r'^(NO|AL|SP|EM|CS|QTY):\s*(.+)$', stripped)
                if m:
                    hdr[m.group(1)] = m.group(2).strip()

        if 'NO' not in hdr:
            continue  # skip malformed / empty blocks

        records.append({
            'test_id':     hdr['NO'],
            'sp':          int(hdr['SP']),
            'em':          int(hdr.get('EM', 0)),
            'cs':          float(hdr.get('CS', 0.0)),
            'qty':         int(hdr.get('QTY', len(seqs))),
            'seqs':        seqs,
            'profile_sp':  compute_profile_score(seqs),
            'computed_cs': compute_cs(seqs),
        })

    # Some RPT files duplicate their entries (e.g. two sections of the same data).
    # Keep only the first occurrence of each test_id.
    seen = set()
    unique = []
    for rec in records:
        if rec['test_id'] not in seen:
            seen.add(rec['test_id'])
            unique.append(rec)
    return unique


# ═══════════════════════════════════════════════════════════════════════════════
#  RPT FORMAT B PARSER  (result/reportDPAMSA_GA/)
#
#  Structure (each entry separated by a blank line):
#      Dataset name: testN
#      SP: -216
#      Alignment:
#      <seq1>
#      <seq2>
#      ...
#
#  Files may contain a second section after
#  "NOW RANDOM AND NOT IN THE WORST ALIGNED"; that section is discarded.
# ═══════════════════════════════════════════════════════════════════════════════

def parse_ga_rpt(path: str) -> dict:
    """Return dict {test_id -> {sp, seqs, profile_sp, computed_cs}}.
    Only the first (best) section is kept."""
    with open(path) as fh:
        content = fh.read()

    # drop the random/worst section
    marker = "NOW RANDOM AND NOT IN THE WORST ALIGNED"
    if marker in content:
        content = content[:content.index(marker)]

    results = {}
    blocks = [b.strip() for b in re.split(r'\n\s*\n', content) if b.strip()]

    for block in blocks:
        lines = block.splitlines()
        test_id = None
        sp = None
        seqs = []
        reading_seqs = False

        for line in lines:
            stripped = line.strip()
            m_name = re.match(r'^Dataset name:\s*(.+)$', stripped)
            m_sp   = re.match(r'^SP:\s*(-?\d+)$', stripped)
            if m_name:
                test_id = m_name.group(1).strip()
            elif m_sp:
                sp = int(m_sp.group(1))
            elif stripped == 'Alignment:':
                reading_seqs = True
            elif reading_seqs and stripped:
                seqs.append(stripped)

        if test_id is not None and sp is not None:
            results[test_id] = {
                'sp':          sp,
                'seqs':        seqs,
                'profile_sp':  compute_profile_score(seqs),
                'computed_cs': compute_cs(seqs),
            }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise_tid(raw: str) -> str:
    """Strip .fasta suffix and whitespace from test-ID strings."""
    return re.sub(r'\.fasta$', '', str(raw).strip())


def parse_ga_dpamsa_csv(path: str) -> pd.DataFrame:
    """*_GA-DPAMSA.py.csv → tidy long-format DataFrame with method column."""
    df = pd.read_csv(path)
    df['test_id'] = df.iloc[:, 0].apply(_normalise_tid)
    rows = []
    for _, row in df.iterrows():
        rows.append({'test_id': row['test_id'], 'method': 'DPAMSA',    'sp': row['DPAMSA_SP']})
        rows.append({'test_id': row['test_id'], 'method': 'GA-DPAMSA', 'sp': row['GA_SP']})
    return pd.DataFrame(rows)


def parse_clustal_csv(path: str) -> pd.DataFrame:
    """*_clustalo_msaprobs_clustalw.csv → tidy long-format DataFrame."""
    df = pd.read_csv(path)
    df.iloc[:, 0] = df.iloc[:, 0].apply(_normalise_tid)
    df.rename(columns={df.columns[0]: 'test_id'}, inplace=True)

    col_map = {
        'SP_ClustalOmega': 'ClustalOmega',
        'SP_MSAProbs':     'MSAProbs',
        'SP_ClustalW':     'ClustalW',
    }
    rows = []
    for _, row in df.iterrows():
        for csv_col, method in col_map.items():
            if csv_col not in df.columns:
                continue
            val = row[csv_col]
            if pd.notna(val) and str(val).strip() not in ('', 'nan'):
                try:
                    rows.append({'test_id': row['test_id'], 'method': method, 'sp': int(float(val))})
                except (ValueError, TypeError):
                    pass
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  MASTER TABLE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

# Datasets: which files supply which data
_DATASETS = [
    dict(
        key    = 'dataset1_3x30bp',
        rpt_a  = os.path.join(RPT_DIR_A, 'dataset1_3x30bp.py.rpt'),
        rpt_b  = os.path.join(RPT_DIR_B, 'dataset1_3x30bp.py.rpt'),
        ga_csv = os.path.join(BENCH_DIR,  'dataset1_3x30bp_GA-DPAMSA.py.csv'),
        cl_csv = os.path.join(BENCH_DIR,  'dataset1_3x30bp_clustalo_msaprobs_clustalw.csv'),
    ),
    dict(
        key    = 'dataset1_6x30bp',
        rpt_a  = os.path.join(RPT_DIR_A, 'dataset1_6x30bp.py.rpt'),
        rpt_b  = os.path.join(RPT_DIR_B, 'dataset1_6x30bp.py.rpt'),
        ga_csv = os.path.join(BENCH_DIR,  'dataset1_6x30bp_GA-DPAMSA.py.csv'),
        cl_csv = os.path.join(BENCH_DIR,  'dataset1_6x30bp_clustalo_msaprobs_clustalw.csv'),
    ),
    dict(
        key    = 'dataset1_6x60bp',
        rpt_a  = None,  # DPAMSA agent did not run on 6×60 bp
        rpt_b  = os.path.join(RPT_DIR_B, 'dataset1_6x60bp.py.rpt'),
        ga_csv = None,  # no GA-DPAMSA CSV for 6×60 bp
        cl_csv = os.path.join(BENCH_DIR,  'dataset1_6x60bp_clustalo_msaprobs_clustalw.csv'),
    ),
]


def build_master_table() -> pd.DataFrame:
    """Assemble a unified DataFrame with columns:
       dataset | test_id | method | sp | cs | computed_cs | profile_sp
    """
    frames = []

    for ds in _DATASETS:
        key = ds['key']

        # ── (1) GA-DPAMSA CSV: DPAMSA and GA-DPAMSA SP scores ────────────────
        if ds['ga_csv'] and os.path.exists(ds['ga_csv']):
            df = parse_ga_dpamsa_csv(ds['ga_csv'])
            df.insert(0, 'dataset', key)
            frames.append(df)

        # ── (2) Clustal-family CSV: ClustalOmega / MSAProbs / ClustalW ───────
        if ds['cl_csv'] and os.path.exists(ds['cl_csv']):
            df = parse_clustal_csv(ds['cl_csv'])
            df.insert(0, 'dataset', key)
            frames.append(df)

        # ── (3) Format-A RPT: DPAMSA alignments ──────────────────────────────
        #   adds 'DPAMSA Profile' rows  (profile SP == SP, but computed independently)
        #   and preserves the original 'cs' reported by the agent
        if ds['rpt_a'] and os.path.exists(ds['rpt_a']):
            recs = parse_dpamsa_rpt(ds['rpt_a'])
            for rec in recs:
                frames.append(pd.DataFrame([{
                    'dataset':     key,
                    'test_id':     rec['test_id'],
                    'method':      'DPAMSA Profile',
                    'sp':          rec['profile_sp'],
                    'cs':          rec['cs'],
                    'computed_cs': rec['computed_cs'],
                }]))
                # also keep the agent-reported SP under a separate column
                # by injecting it as a separate 'rpt_sp' row into the DPAMSA method
                # (used only in fig3)

        # ── (4) Format-B RPT: GA-DPAMSA alignments ───────────────────────────
        #   These rows carry computed_cs and the RPT-reported SP.
        #   For datasets where GA CSV already exists (3x30, 6x30), we keep them
        #   only for cs and profile data; the SP from CSV is preferred.
        #   For 6x60 (no GA CSV), these rows ARE the GA-DPAMSA SP source.
        if ds['rpt_b'] and os.path.exists(ds['rpt_b']):
            ga_recs = parse_ga_rpt(ds['rpt_b'])
            has_csv_ga = ds['ga_csv'] and os.path.exists(ds['ga_csv'])
            for tid, rec in ga_recs.items():
                row: dict = {
                    'dataset':     key,
                    'test_id':     tid,
                    'sp':          rec['sp'],
                    'computed_cs': rec['computed_cs'],
                }
                if has_csv_ga:
                    # supplement-only: expose profile & CS, avoid duplicate SP bars
                    row['method'] = 'GA-DPAMSA Profile'
                else:
                    # primary source for this dataset's GA-DPAMSA SP
                    row['method'] = 'GA-DPAMSA'
                frames.append(pd.DataFrame([row]))

    if not frames:
        return pd.DataFrame(columns=['dataset', 'test_id', 'method', 'sp',
                                     'cs', 'computed_cs'])

    master = pd.concat(frames, ignore_index=True, sort=False)
    master['sp'] = pd.to_numeric(master['sp'], errors='coerce')
    return master


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _color(method: str) -> str:
    for k, v in METHOD_COLORS.items():
        if k.lower() in method.lower():
            return v
    return "#607D8B"  # blue-grey fallback


def _order(methods):
    """Return methods in a consistent, preferred display order."""
    present = set(methods)
    ordered = [m for m in _DISPLAY_ORDER if m in present]
    ordered += sorted(present - set(ordered))
    return ordered


def _save(fig, outdir: str, fname: str):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"    saved → {path}")
    plt.close(fig)


def _test_num(tid: str) -> int:
    m = re.search(r'(\d+)$', str(tid))
    return int(m.group(1)) if m else -1


def _primary_methods(master: pd.DataFrame, dataset=None) -> list:
    """Return comparison-relevant methods (exclude *Profile and *rpt variants)."""
    sub = master if dataset is None else master[master['dataset'] == dataset]
    exclude = {'DPAMSA Profile', 'GA-DPAMSA Profile'}
    return _order([m for m in sub['method'].unique() if m not in exclude])


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Grouped Bar: Mean SP per Dataset per Method
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_mean_sp_bar(master: pd.DataFrame, outdir: str):
    print("  fig1: mean SP bar chart …")
    datasets = list(DATASET_LABELS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Mean SP Score by Algorithm and Dataset\n"
                 "(higher = better; error bars = ±1 SEM)",
                 fontsize=13, fontweight='bold')

    for ax, ds_key in zip(axes, datasets):
        methods = _primary_methods(master, ds_key)
        sub     = master[master['dataset'] == ds_key]
        means   = [sub[sub['method'] == m]['sp'].mean() for m in methods]
        sems    = [sub[sub['method'] == m]['sp'].sem()  for m in methods]
        colors  = [_color(m) for m in methods]

        x    = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=sems, color=colors, alpha=0.85,
                      capsize=4, error_kw={'linewidth': 1.2, 'ecolor': '#333'})

        ax.set_title(DATASET_LABELS[ds_key], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=38, ha='right', fontsize=8.5)
        ax.set_ylabel("Mean SP Score" if ax is axes[0] else "")
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.grid(axis='y', alpha=0.3)

        for bar, mean in zip(bars, means):
            if not np.isnan(mean):
                y_off = abs(mean) * 0.02
                ax.text(bar.get_x() + bar.get_width() / 2,
                        mean - y_off,
                        f"{mean:.0f}", ha='center', va='top', fontsize=7.5)

    fig.tight_layout()
    _save(fig, outdir, "fig1_mean_sp_bar.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Violin: SP Distribution per Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_violin_distribution(master: pd.DataFrame, outdir: str):
    print("  fig2: violin distribution …")
    datasets = list(DATASET_LABELS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("SP Score Distribution by Algorithm\n(each violin = 50 test cases)",
                 fontsize=13, fontweight='bold')

    for ax, ds_key in zip(axes, datasets):
        methods = _primary_methods(master, ds_key)
        sub     = master[master['dataset'] == ds_key].dropna(subset=['sp'])
        data    = [sub[sub['method'] == m]['sp'].values for m in methods]
        colors  = [_color(m) for m in methods]

        parts = ax.violinplot(data, positions=range(len(methods)),
                              showmedians=True, showextrema=True)
        for pc, col in zip(parts['bodies'], colors):
            pc.set_facecolor(col)
            pc.set_alpha(0.72)
        for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
            if key in parts:
                parts[key].set_color('#333')
                parts[key].set_linewidth(1.1)

        # overlay individual data points
        for i, vals in enumerate(data):
            ax.scatter(np.full(len(vals), i), vals,
                       color=colors[i], alpha=0.35, s=8, zorder=3)

        ax.set_title(DATASET_LABELS[ds_key], fontsize=11)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=38, ha='right', fontsize=8.5)
        ax.set_ylabel("SP Score" if ax is axes[0] else "")
        ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    _save(fig, outdir, "fig2_violin_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — SP vs Profile Score Scatter  (validation)
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_sp_vs_profile(master: pd.DataFrame, outdir: str):
    """Show that Profile SP (computed) == SP from the DPAMSA agent's CSV/RPT."""
    print("  fig3: SP vs profile scatter (validation) …")

    dpamsa_sp  = (master[master['method'] == 'DPAMSA']
                  [['dataset', 'test_id', 'sp']]
                  .rename(columns={'sp': 'sp_csv'}))
    dpamsa_prf = (master[master['method'] == 'DPAMSA Profile']
                  [['dataset', 'test_id', 'sp']]
                  .rename(columns={'sp': 'profile_sp'}))

    merged = pd.merge(dpamsa_sp, dpamsa_prf, on=['dataset', 'test_id'])
    if merged.empty:
        print("    [skip] No overlapping DPAMSA / DPAMSA Profile rows.")
        return

    datasets = [d for d in DATASET_LABELS if d in merged['dataset'].unique()]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5.5))
    if len(datasets) == 1:
        axes = [axes]
    fig.suptitle("SP Score (from CSV) vs Profile Score (re-computed from alignment)\n"
                 "Both should lie on the y = x identity line",
                 fontsize=12, fontweight='bold')

    for ax, ds_key in zip(axes, datasets):
        sub = merged[merged['dataset'] == ds_key]
        ax.scatter(sub['sp_csv'], sub['profile_sp'],
                   color=METHOD_COLORS['DPAMSA'], alpha=0.75, s=60,
                   edgecolors='white', linewidths=0.6)

        lo = min(sub['sp_csv'].min(), sub['profile_sp'].min()) * 1.06
        hi = max(sub['sp_csv'].max(), sub['profile_sp'].max()) * 0.94
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.2, label='y = x (perfect match)')

        ax.set_xlabel("SP Score (DPAMSA CSV)", fontsize=10)
        ax.set_ylabel("Profile SP (computed from RPT alignment)", fontsize=10)
        ax.set_title(DATASET_LABELS[ds_key])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        corr  = sub['sp_csv'].corr(sub['profile_sp'])
        n_eq  = (sub['sp_csv'] == sub['profile_sp']).sum()
        ax.text(0.05, 0.95,
                f"r = {corr:.4f}\nexact match: {n_eq}/{len(sub)}",
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    fig.tight_layout()
    _save(fig, outdir, "fig3_sp_vs_profile_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Per-Test Line Plots (test0 … test49)
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_per_test_lines(master: pd.DataFrame, outdir: str):
    print("  fig4: per-test line plots …")
    datasets = list(DATASET_LABELS.keys())
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle("SP Score per Test Case  (test0 – test49)\n"
                 "All methods shown on the same scale per dataset",
                 fontsize=13, fontweight='bold')

    for ax, ds_key in zip(axes, datasets):
        sub     = master[master['dataset'] == ds_key].dropna(subset=['sp'])
        methods = _primary_methods(master, ds_key)

        for method in methods:
            mdf = (sub[sub['method'] == method]
                   .copy()
                   .assign(test_num=lambda d: d['test_id'].apply(_test_num))
                   .sort_values('test_num'))
            ax.plot(mdf['test_num'], mdf['sp'],
                    label=method, color=_color(method),
                    linewidth=1.5, alpha=0.85, marker='o', markersize=3)

        ax.set_title(DATASET_LABELS[ds_key], fontsize=11)
        ax.set_xlabel("Test index")
        ax.set_ylabel("SP Score")
        ax.legend(fontsize=8.5, ncol=3, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xlim(-1, 50)

    fig.tight_layout()
    _save(fig, outdir, "fig4_per_test_lines.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Win-Rate Heatmap  (row beats column)
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_win_rate_heatmap(master: pd.DataFrame, outdir: str):
    """For each ordered method pair (A, B), compute P(SP_A > SP_B)."""
    print("  fig5: win-rate heatmap …")
    datasets = list(DATASET_LABELS.keys())
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5))
    fig.suptitle("Win Rate: P(row method SP > column method SP)\n"
                 "Green = row wins more often; Red = row loses more often",
                 fontsize=12, fontweight='bold')

    for ax, ds_key in zip(axes, datasets):
        sub     = master[master['dataset'] == ds_key].dropna(subset=['sp'])
        methods = _primary_methods(master, ds_key)
        wide    = sub.pivot_table(index='test_id', columns='method',
                                  values='sp', aggfunc='first')
        # keep only methods present in this dataset
        methods = [m for m in methods if m in wide.columns]
        wide    = wide[methods]

        nm = len(methods)
        win = np.full((nm, nm), np.nan)
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                if i == j:
                    win[i, j] = np.nan
                    continue
                both = wide[[m1, m2]].dropna()
                if len(both) > 0:
                    win[i, j] = (both[m1] > both[m2]).mean()

        im = ax.imshow(win, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(nm)); ax.set_yticks(range(nm))
        ax.set_xticklabels(methods, rotation=40, ha='right', fontsize=8.5)
        ax.set_yticklabels(methods, fontsize=8.5)
        ax.set_title(DATASET_LABELS[ds_key], fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Win rate')

        for i in range(nm):
            for j in range(nm):
                if not np.isnan(win[i, j]):
                    text_col = 'white' if abs(win[i, j] - 0.5) > 0.3 else 'black'
                    ax.text(j, i, f"{win[i,j]:.2f}",
                            ha='center', va='center', fontsize=8, color=text_col)

    fig.tight_layout()
    _save(fig, outdir, "fig5_win_rate_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — Relative SP Improvement over DPAMSA Baseline
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_relative_improvement(master: pd.DataFrame, outdir: str):
    """(method_SP − DPAMSA_SP) / |DPAMSA_SP| × 100  averaged over 50 tests."""
    print("  fig6: relative improvement bar chart …")
    datasets = list(DATASET_LABELS.keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(16, 5.5))
    fig.suptitle("Mean SP Improvement Relative to DPAMSA Baseline (%)\n"
                 "Positive = better than DPAMSA; negative = worse",
                 fontsize=12, fontweight='bold')

    for ax, ds_key in zip(axes, datasets):
        sub = master[master['dataset'] == ds_key].dropna(subset=['sp'])

        # choose baseline
        if 'DPAMSA' in sub['method'].values:
            base_method = 'DPAMSA'
        elif 'GA-DPAMSA' in sub['method'].values:
            base_method = 'GA-DPAMSA'
        else:
            ax.set_title(DATASET_LABELS[ds_key] + "\n(no baseline)")
            continue

        baseline = (sub[sub['method'] == base_method][['test_id', 'sp']]
                    .rename(columns={'sp': 'base'}))

        methods = [m for m in _primary_methods(master, ds_key) if m != base_method]
        improvements, errors = [], []
        for method in methods:
            mdf    = sub[sub['method'] == method][['test_id', 'sp']]
            merged = pd.merge(mdf, baseline, on='test_id')
            if merged.empty:
                improvements.append(np.nan)
                errors.append(0)
            else:
                rel = (merged['sp'] - merged['base']) / merged['base'].abs() * 100
                improvements.append(rel.mean())
                errors.append(rel.sem())

        colors = [_color(m) for m in methods]
        x      = np.arange(len(methods))
        bars   = ax.bar(x, improvements, yerr=errors, color=colors, alpha=0.85,
                        capsize=4, error_kw={'linewidth': 1.2, 'ecolor': '#333'})

        ax.axhline(0, color='black', linewidth=1.0)
        ax.set_title(f"{DATASET_LABELS[ds_key]}\n(baseline: {base_method})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=38, ha='right', fontsize=8.5)
        ax.set_ylabel("Relative improvement (%)" if ax is axes[0] else "")
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, improvements):
            if not np.isnan(val):
                off = 0.6 if val >= 0 else -1.2
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + off, f"{val:+.1f}%",
                        ha='center', fontsize=7.5)

    fig.tight_layout()
    _save(fig, outdir, "fig6_relative_improvement.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 7 — Column Score (CS) — DPAMSA and GA-DPAMSA alignments
# ═══════════════════════════════════════════════════════════════════════════════

def fig7_column_score(master: pd.DataFrame, outdir: str):
    print("  fig7: column score …")
    # Gather CS data: agent-reported (Format A) and computed
    cs_data = master.dropna(subset=['cs'])
    cc_data = master.dropna(subset=['computed_cs'])

    if cs_data.empty and cc_data.empty:
        print("    [skip] No CS data found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Column Score (CS) — Fraction of Perfectly-Conserved Columns",
                 fontsize=13, fontweight='bold')

    # ── Left: reported CS from Format A (DPAMSA agent) ───────────────────────
    ax = axes[0]
    for ds_key, label in DATASET_LABELS.items():
        sub = cs_data[cs_data['dataset'] == ds_key].copy()
        if sub.empty:
            continue
        sub['tn'] = sub['test_id'].apply(_test_num)
        sub = sub.sort_values('tn')
        ax.plot(sub['tn'], sub['cs'],
                label=label, linewidth=1.4, marker='o', markersize=3)
    ax.set_title("Reported CS  (DPAMSA agent, Format-A RPT)")
    ax.set_xlabel("Test index")
    ax.set_ylabel("Column Score")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, 1.02)

    # ── Right: computed CS for DPAMSA and GA-DPAMSA ───────────────────────────
    ax = axes[1]
    for ds_key, d_label in DATASET_LABELS.items():
        sub = cc_data[cc_data['dataset'] == ds_key]
        for method in _order(sub['method'].unique()):
            ms = sub[sub['method'] == method].copy()
            if ms.empty:
                continue
            ms['tn'] = ms['test_id'].apply(_test_num)
            ms = ms.sort_values('tn')
            ax.plot(ms['tn'], ms['computed_cs'],
                    label=f"{method}\n({d_label})",
                    linewidth=1.4, marker='o', markersize=3,
                    color=_color(method), alpha=0.85,
                    linestyle='-' if 'DPAMSA' in method and 'GA' not in method else '--')
    ax.set_title("Computed CS  (from alignment text in RPT files)")
    ax.set_xlabel("Test index")
    ax.set_ylabel("Column Score")
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    _save(fig, outdir, "fig7_column_score.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 8 — Summary Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def fig8_summary_dashboard(master: pd.DataFrame, outdir: str):
    print("  fig8: summary dashboard …")
    fig = plt.figure(figsize=(15, 7), layout='constrained')
    fig.suptitle("MSA Algorithm Comparison — Summary Dashboard",
                 fontsize=14, fontweight='bold')
    gs = GridSpec(1, 2, figure=fig, wspace=0.38, left=0.04, right=0.96)

    ax_tbl = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    # ── Mean-SP summary table ─────────────────────────────────────────────────
    primary = master[~master['method'].str.contains('Profile', na=False)]
    pivot   = (primary.groupby(['dataset', 'method'])['sp']
                .mean()
                .reset_index())
    pivot['dataset'] = pivot['dataset'].map(DATASET_LABELS).fillna(pivot['dataset'])
    tbl_df  = pivot.pivot(index='method', columns='dataset', values='sp').round(0)
    tbl_df  = tbl_df.reindex([m for m in _DISPLAY_ORDER if m in tbl_df.index])

    ax_tbl.axis('off')
    col_labels = tbl_df.columns.tolist()
    row_labels = tbl_df.index.tolist()
    cell_text  = [
        [f"{v:.0f}" if not np.isnan(v) else "—" for v in tbl_df.loc[r]]
        for r in row_labels
    ]
    tbl = ax_tbl.table(
        cellText  = cell_text,
        rowLabels = row_labels,
        colLabels = col_labels,
        loc       = 'center',
        cellLoc   = 'center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.25, 1.7)
    # colour row headers
    for (r_idx, c_idx), cell in tbl.get_celld().items():
        if r_idx == 0 or c_idx == -1:
            if r_idx == 0:
                cell.set_facecolor('#CFD8DC')
            else:
                method_name = row_labels[r_idx - 1]
                cell.set_facecolor(
                    mcolors.to_rgba(_color(method_name), alpha=0.25))
    ax_tbl.set_title("Mean SP Score  (higher = better)", pad=14, fontsize=10.5)

    # ── Grouped bar: all methods × all datasets ───────────────────────────────
    all_methods = [m for m in _DISPLAY_ORDER
                   if m in primary['method'].unique()]
    datasets   = list(DATASET_LABELS.keys())
    d_labels   = list(DATASET_LABELS.values())

    x     = np.arange(len(all_methods))
    width = 0.27
    cmap  = plt.get_cmap('tab10')

    for di, (ds_key, d_label) in enumerate(zip(datasets, d_labels)):
        sub = primary[primary['dataset'] == ds_key]
        means = [
            sub[sub['method'] == m]['sp'].mean()
            if m in sub['method'].values else np.nan
            for m in all_methods
        ]
        ax_bar.bar(x + di * width, means, width,
                   label=d_label, color=cmap(di), alpha=0.82)

    ax_bar.set_xticks(x + width)
    ax_bar.set_xticklabels(all_methods, rotation=38, ha='right', fontsize=9)
    ax_bar.set_ylabel("Mean SP Score")
    ax_bar.set_title("All Methods × All Datasets (side-by-side)", fontsize=10.5)
    ax_bar.legend(fontsize=9, loc='lower right')
    ax_bar.grid(axis='y', alpha=0.3)
    ax_bar.axhline(0, color='black', linewidth=0.5)

    _save(fig, outdir, "fig8_summary_dashboard.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MSA Algorithm Comparison Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--outdir", default=os.path.join(ROOT, "figures"),
        help="Output directory for figures  (default: ./figures)"
    )
    parser.add_argument(
        "--csv", default=os.path.join(ROOT, "results", "comparison_summary.csv"),
        help="Path for the summary CSV export"
    )
    args = parser.parse_args()

    print("=" * 64)
    print("  MSA Algorithm Comparison Pipeline")
    print("=" * 64)

    # ── Step 1: build master table ────────────────────────────────────────────
    print("\n[1/3]  Building master comparison table …")
    master = build_master_table()
    print(f"       {len(master):,} rows loaded")
    print(f"       methods  : {sorted(master['method'].unique())}")
    print(f"       datasets : {sorted(master['dataset'].unique())}")

    # ── Step 2: export CSV summary ────────────────────────────────────────────
    print("\n[2/3]  Exporting summary CSV …")
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    summary = (
        master
        .groupby(['dataset', 'method'])['sp']
        .agg(count='count', mean='mean', std='std',
             median='median', min='min', max='max')
        .reset_index()
        .round(2)
    )
    summary.to_csv(args.csv, index=False)
    print(f"       saved → {args.csv}")
    print()
    print(summary.to_string(index=False))

    # ── Step 3: generate figures ──────────────────────────────────────────────
    print("\n[3/3]  Generating figures …")
    fig1_mean_sp_bar(master, args.outdir)
    fig2_violin_distribution(master, args.outdir)
    fig3_sp_vs_profile(master, args.outdir)
    fig4_per_test_lines(master, args.outdir)
    fig5_win_rate_heatmap(master, args.outdir)
    fig6_relative_improvement(master, args.outdir)
    fig7_column_score(master, args.outdir)
    fig8_summary_dashboard(master, args.outdir)

    print(f"\nAll done!  Figures → {args.outdir}")
    print(f"           Summary → {args.csv}")


if __name__ == "__main__":
    main()
