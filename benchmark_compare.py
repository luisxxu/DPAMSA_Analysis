"""
benchmark_compare.py
====================
Utilities for comparing an ACER (or any RL agent) SP score against the
pre-computed classical-tool benchmark stored in datasets/benchmark/*.csv.

Public API
----------
  lookup(fasta_path, agent_score, episodes, time_s)
      → dict  — full comparison record for one run

  print_comparison(record)
      → None  — pretty-print a single-run comparison table to stdout

  append_results(record, csv_path)
      → None  — append / create the cumulative results CSV

  generate_figures(csv_path, figures_dir)
      → list[str]  — save comparison figures, return their paths

Dataset detection
-----------------
The FASTA file path must contain one of the recognised dataset directory
names (dataset1_3x30bp, dataset1_6x30bp, dataset1_6x60bp) so the correct
benchmark CSV can be located automatically.

Benchmark CSV format (datasets/benchmark/)
-----------------------------------------
  3x30bp :  File name = "test0"         (no extension), ClustalW = empty
  6x30bp :  File name = "test0.fasta"   (with extension)
  6x60bp :  File name = "test0.fasta"   (with extension)
"""

from __future__ import annotations

import os
import csv
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── constants ─────────────────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Maps a dataset directory fragment → benchmark CSV filename
_BENCHMARK_CSVS: dict[str, str] = {
    "dataset1_3x30bp": "dataset1_3x30bp_clustalo_msaprobs_clustalw.csv",
    "dataset1_6x30bp": "dataset1_6x30bp_clustalo_msaprobs_clustalw.csv",
    "dataset1_6x60bp": "dataset1_6x60bp_clustalo_msaprobs_clustalw.csv",
}

# Display names for each method (column order for plots)
# _CLASSICAL_TOOLS: scores loaded from pre-computed benchmark CSVs
# _RUNTIME_TOOLS:   scores computed live during a run (e.g. MAFFT)
_CLASSICAL_TOOLS = ["SP_ClustalOmega", "SP_MSAProbs", "SP_ClustalW"]
_RUNTIME_TOOLS   = ["SP_MAFFT"]
_ALL_TOOLS       = _CLASSICAL_TOOLS + _RUNTIME_TOOLS   # used for display / figures

_TOOL_LABELS     = {
    "SP_ACER":         "ACER",
    "SP_ClustalOmega": "ClustalOmega",
    "SP_MSAProbs":     "MSAProbs",
    "SP_ClustalW":     "ClustalW",
    "SP_MAFFT":        "MAFFT",
}
_TOOL_COLORS = {
    "SP_ACER":         "#9C27B0",   # purple
    "SP_ClustalOmega": "#2196F3",   # blue
    "SP_MSAProbs":     "#4CAF50",   # green
    "SP_ClustalW":     "#FF9800",   # orange
    "SP_MAFFT":        "#F44336",   # red
}

# Columns saved in the cumulative results CSV
_CSV_COLS = ["dataset", "test_id", "episodes", "time_s",
             "SP_ACER", "SP_ClustalOmega", "SP_MSAProbs", "SP_ClustalW", "SP_MAFFT"]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _detect_dataset(fasta_path: str) -> tuple[str, str]:
    """Return (dataset_key, test_stem) parsed from *fasta_path*.

    Raises ValueError if the path does not match a known dataset.
    """
    norm = fasta_path.replace("\\", "/")
    for key in _BENCHMARK_CSVS:
        if key in norm:
            # test_stem = basename without extension, e.g. "test0"
            basename  = os.path.basename(norm)
            test_stem = os.path.splitext(basename)[0]
            return key, test_stem
    raise ValueError(
        f"Cannot detect dataset from path '{fasta_path}'.\n"
        f"Path must contain one of: {list(_BENCHMARK_CSVS)}")


def _load_benchmark_row(dataset_key: str, test_stem: str) -> dict[str, float | None]:
    """Return classical SP scores for *test_stem* from the benchmark CSV.

    Handles both naming conventions in the CSVs:
      "test0"        (3x30bp — no .fasta suffix)
      "test0.fasta"  (6x30bp, 6x60bp — with suffix)
    """
    csv_name = _BENCHMARK_CSVS[dataset_key]
    csv_path = os.path.join(_PROJECT_ROOT, "datasets", "benchmark", csv_name)

    df = pd.read_csv(csv_path)
    # Normalise the File name column to stem (strip .fasta if present)
    df["_stem"] = df.iloc[:, 0].astype(str).str.replace(r"\.fasta$", "",
                                                          regex=True).str.strip()
    match = df[df["_stem"] == test_stem]
    if match.empty:
        raise KeyError(
            f"Test ID '{test_stem}' not found in {csv_name}. "
            f"Available: {df['_stem'].tolist()}")

    row = match.iloc[0]
    result: dict[str, float | None] = {}
    for col in _CLASSICAL_TOOLS:
        val = row.get(col, float("nan"))
        try:
            f = float(val)
            result[col] = None if math.isnan(f) else f
        except (TypeError, ValueError):
            result[col] = None
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def lookup(fasta_path: str, agent_score: float,
           episodes: int = 0, time_s: float = 0.0,
           mafft_score: "float | None" = None) -> dict:
    """Build a full comparison record for one training run.

    Parameters
    ----------
    fasta_path  : path to the FASTA file that was aligned
    agent_score : SP score returned by the RL agent after inference
    episodes    : number of training episodes (for the record)
    time_s      : training wall-clock time in seconds
    mafft_score : SP score from MAFFT (computed live); None if unavailable

    Returns
    -------
    dict with keys: dataset, test_id, episodes, time_s,
                    SP_ACER, SP_ClustalOmega, SP_MSAProbs, SP_ClustalW, SP_MAFFT
    """
    dataset_key, test_stem = _detect_dataset(fasta_path)
    bench = _load_benchmark_row(dataset_key, test_stem)

    # Pretty dataset tag for display (e.g. "3x30bp")
    dataset_tag = dataset_key.replace("dataset1_", "")

    return {
        "dataset":         dataset_tag,
        "test_id":         test_stem,
        "episodes":        episodes,
        "time_s":          round(time_s, 2),
        "SP_ACER":         agent_score,
        "SP_ClustalOmega": bench["SP_ClustalOmega"],
        "SP_MSAProbs":     bench["SP_MSAProbs"],
        "SP_ClustalW":     bench["SP_ClustalW"],
        "SP_MAFFT":        mafft_score,
    }


def print_comparison(record: dict) -> None:
    """Pretty-print a single-run benchmark comparison table."""
    dataset  = record["dataset"]
    test_id  = record["test_id"]
    episodes = record["episodes"]

    # Collect all valid (method, score) pairs
    methods = [("ACER", record["SP_ACER"])]
    for col in _ALL_TOOLS:
        val = record.get(col)
        if val is not None:
            methods.append((_TOOL_LABELS[col], val))

    # Sort best → worst (higher SP is better)
    methods_sorted = sorted(methods, key=lambda x: x[1], reverse=True)
    best_score = methods_sorted[0][1]

    # Find ACER rank
    acer_rank = next(i + 1 for i, (m, _) in enumerate(methods_sorted)
                     if m == "ACER")

    title = f"Benchmark Comparison — {dataset} / {test_id}  ({episodes} episodes)"
    width = max(len(title) + 4, 58)
    sep   = "─" * width

    print(f"\n┌{sep}┐")
    print(f"│  {title:<{width - 2}}│")
    print(f"├{'─'*18}┬{'─'*10}┬{'─'*12}┬{'─'*8}┤")
    print(f"│ {'Method':<16} │ {'SP Score':>8} │ {'vs ACER':>10} │ {'Rank':>6} │")
    print(f"├{'─'*18}┼{'─'*10}┼{'─'*12}┼{'─'*8}┤")

    acer_score = record["SP_ACER"]
    for rank, (name, score) in enumerate(methods_sorted, 1):
        star   = "★ " if name == "ACER" else "  "
        delta  = "" if name == "ACER" else f"{score - acer_score:+.0f}"
        rank_s = str(rank)
        print(f"│ {star}{name:<15} │ {score:>8.0f} │ {delta:>10} │ {rank_s:>6} │")

    print(f"└{'─'*18}┴{'─'*10}┴{'─'*12}┴{'─'*8}┘")

    # Summary line
    n_tools = len(methods) - 1  # excluding ACER
    if acer_rank == 1:
        verdict = f"ACER is BEST (rank 1 / {len(methods)})"
    else:
        beaten_by = [m for m, s in methods_sorted[:acer_rank - 1]]
        verdict   = (f"ACER is rank {acer_rank} / {len(methods)} "
                     f"(beaten by: {', '.join(beaten_by)})")
    print(f"  → {verdict}\n")


def append_results(record: dict, csv_path: str) -> None:
    """Upsert *record* into the cumulative results CSV.

    If a row with the same (dataset, test_id) already exists it is
    **replaced** with the new record, so re-running a test always
    reflects the most recent result and duplicate rows never accumulate.
    The CSV is created (with a header) if it does not yet exist.
    """
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    # Load existing data (empty DataFrame if file absent)
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        # Back-fill columns added in later versions
        for col in _CSV_COLS:
            if col not in df.columns:
                df[col] = float("nan")
    else:
        df = pd.DataFrame(columns=_CSV_COLS)

    # Drop any existing rows for this (dataset, test_id) pair
    mask = (df["dataset"] == record["dataset"]) & (df["test_id"] == record["test_id"])
    df = df[~mask]

    # Append the new record and write back
    new_row = pd.DataFrame([{col: record.get(col) for col in _CSV_COLS}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_path, index=False)


def generate_figures(csv_path: str, figures_dir: str) -> list[str]:
    """Generate comparison figures from the cumulative results CSV.

    Figures produced
    ----------------
    bench1_grouped_bar.png  — SP per test case, grouped by method
    bench2_delta.png        — ACER score minus each classical tool per test
    bench3_winrate.png      — fraction of tests ACER beats each classical tool
    bench4_summary_table.png — all SP scores as a formatted table

    Returns list of saved file paths.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    # Back-fill any columns absent from older CSVs (e.g. SP_MAFFT added later)
    for col in _CSV_COLS:
        if col not in df.columns:
            df[col] = float("nan")

    os.makedirs(figures_dir, exist_ok=True)
    saved = []

    # Determine which tools have at least one non-NaN value
    available = [c for c in _ALL_TOOLS if df[c].notna().any()]
    all_methods = ["SP_ACER"] + available
    n_tests = len(df)

    # ── Figure 1: grouped bar chart ───────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(max(10, n_tests * 0.9 + 3), 5))
    x     = np.arange(n_tests)
    n_m   = len(all_methods)
    width = 0.8 / n_m

    for i, col in enumerate(all_methods):
        vals    = df[col].tolist()
        offset  = (i - n_m / 2 + 0.5) * width
        bars    = ax1.bar(x + offset, vals, width,
                          label=_TOOL_LABELS[col],
                          color=_TOOL_COLORS[col], alpha=0.85,
                          edgecolor="white", linewidth=0.6)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["test_id"].tolist(), rotation=45, ha="right",
                        fontsize=8)
    ax1.set_ylabel("SP Score  (higher = better)")
    ax1.set_title(
        f"SP Score per Test Case — ACER vs Classical Tools\n"
        f"({df['dataset'].iloc[0]}  ·  {df['episodes'].iloc[0]} episodes)",
        fontsize=12)
    ax1.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(axis="y", alpha=0.3)
    fig1.tight_layout()
    p1 = os.path.join(figures_dir, "bench1_grouped_bar.png")
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    saved.append(p1)

    # ── Figure 2: delta (ACER minus each classical tool) ─────────────────────
    fig2, ax2 = plt.subplots(figsize=(max(10, n_tests * 0.9 + 3), 5))
    for i, col in enumerate(available):
        delta   = df["SP_ACER"] - df[col]
        offset  = (i - len(available) / 2 + 0.5) * (0.7 / len(available))
        colors_bar = [_TOOL_COLORS[col] if v >= 0 else "#EF5350"
                      for v in delta]
        ax2.bar(x + offset, delta, 0.7 / len(available),
                color=colors_bar, alpha=0.85,
                label=f"ACER − {_TOOL_LABELS[col]}",
                edgecolor="white", linewidth=0.6)

    ax2.axhline(0, color="black", linewidth=1.2, linestyle="-")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["test_id"].tolist(), rotation=45, ha="right",
                        fontsize=8)
    ax2.set_ylabel("ACER SP − Classical SP\n(positive = ACER wins)")
    ax2.set_title("ACER Advantage per Test Case\n"
                  "(bars above zero = ACER outperforms that tool)", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(figures_dir, "bench2_delta.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    saved.append(p2)

    # ── Figure 3: win-rate summary bar ────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    win_rates = []
    labels_wr = []
    colors_wr = []
    for col in available:
        valid = df[["SP_ACER", col]].dropna()
        if len(valid) == 0:
            continue
        wins = (valid["SP_ACER"] > valid[col]).sum()
        win_rates.append(wins / len(valid) * 100)
        labels_wr.append(_TOOL_LABELS[col])
        colors_wr.append(_TOOL_COLORS[col])

    bars3 = ax3.barh(labels_wr, win_rates, color=colors_wr, alpha=0.85,
                     edgecolor="white")
    ax3.axvline(50, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    ax3.set_xlim(0, 100)
    ax3.set_xlabel("% of test cases where ACER wins")
    ax3.set_title(f"ACER Win Rate vs Classical Tools\n"
                  f"({n_tests} test cases, {df['episodes'].iloc[0]} episodes)",
                  fontsize=12)
    for bar, rate in zip(bars3, win_rates):
        ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{rate:.0f}%", va="center", fontsize=10, fontweight="bold")
    ax3.grid(axis="x", alpha=0.3)
    fig3.tight_layout()
    p3 = os.path.join(figures_dir, "bench3_winrate.png")
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    saved.append(p3)

    # ── Figure 4: summary table ───────────────────────────────────────────────
    # Show all scores + rank for each test case
    display_cols  = ["test_id"] + all_methods
    display_names = ["Test"] + [_TOOL_LABELS[c] for c in all_methods]

    def _rank_row(row):
        scores = {c: row[c] for c in all_methods if pd.notna(row[c])}
        sorted_cols = sorted(scores, key=scores.get, reverse=True)
        return {c: sorted_cols.index(c) + 1 if c in sorted_cols else ""
                for c in all_methods}

    ranks = df.apply(_rank_row, axis=1, result_type="expand")

    # Build cell text: "score\n(rank)"
    cell_text = []
    cell_colors = []
    rank_palette = {1: "#A5D6A7", 2: "#FFF9C4", 3: "#FFCC80", 4: "#EF9A9A", 5: "#CE93D8"}

    for _, row in df.iterrows():
        row_text   = [row["test_id"]]
        row_colors = ["#F5F5F5"]
        for col in all_methods:
            val  = row[col]
            rank = ranks.loc[row.name, col]
            if pd.isna(val):
                row_text.append("N/A")
                row_colors.append("#EEEEEE")
            else:
                row_text.append(f"{int(val)}")
                bg = rank_palette.get(rank, "#FFFFFF")
                row_colors.append(bg)
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    n_rows = len(cell_text)
    fig_h  = max(3, 0.35 * n_rows + 1.5)
    fig4, ax4 = plt.subplots(figsize=(len(display_names) * 2.0, fig_h))
    ax4.axis("off")
    tbl = ax4.table(
        cellText=cell_text,
        colLabels=display_names,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    # Colour header row
    for j in range(len(display_names)):
        tbl[0, j].set_facecolor("#424242")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Legend for rank colours
    legend_patches = [
        mpatches.Patch(color=rank_palette[1], label="Rank 1 (best)"),
        mpatches.Patch(color=rank_palette[2], label="Rank 2"),
        mpatches.Patch(color=rank_palette[3], label="Rank 3"),
        mpatches.Patch(color=rank_palette[4], label="Rank 4"),
        mpatches.Patch(color=rank_palette[5], label="Rank 5"),
    ]
    ax4.legend(handles=legend_patches, loc="upper right",
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(1, 1.05))

    dataset_tag = df["dataset"].iloc[0]
    ep          = df["episodes"].iloc[0]
    ax4.set_title(f"SP Score Summary — {dataset_tag}  ·  {ep} episodes\n"
                  f"(cells shaded by rank; ACER = purple column)",
                  fontsize=11, pad=12, fontweight="bold")
    fig4.tight_layout()
    p4 = os.path.join(figures_dir, "bench4_summary_table.png")
    fig4.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    saved.append(p4)

    return saved
