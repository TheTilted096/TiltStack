#!/usr/bin/env python3
"""Visualize flop bucket label distribution and centroid structure.

Reads the binary flop_labels.bin (1.28M uint16 cluster labels) and
optionally flop_centroids.npy (Kx256 float32) to produce diagnostic
figures plus console summary statistics.

Flop centroids are 256-dim CDF vectors over wide turn buckets (raw counts,
NOT divided by 47.0, so values range from 0 to 47).  Expected EHS is computed
by converting each CDF centroid back to a PDF via finite differences, then
computing (pdf @ wide_turn_ehs) / (47.0 * 46.0 * 255.0), where wide_turn_ehs
(in [0, 46*255] scale) is derived from turn_centroids.npy and river_centroids.npy.
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent.parent.parent / "clusters"
LABELS_PATH = OUTPUT_DIR / "flop_labels.bin"
CENTROIDS_PATH = OUTPUT_DIR / "flop_centroids.npy"
EHS_PATH = OUTPUT_DIR / "flop_ehs.bin"
OUTPUT_PATH = OUTPUT_DIR / "flop_labels_viz.png"
K = 2048


# ── Data loading ─────────────────────────────────────────────────────
def load_label_counts(path, k, chunk=1_000_000):
    """Memory-map labels and compute per-cluster counts in chunks."""
    labels = np.memmap(path, dtype=np.uint16, mode="r")
    n = len(labels)
    counts = np.zeros(k, dtype=np.int64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        counts += np.bincount(labels[start:end], minlength=k)
    return counts, n


def load_counts_and_examples(
    path, k, cluster_ids, n_examples=5, seed=None, chunk=1_000_000
):
    """Single-pass scan: compute per-cluster counts and collect reservoir samples."""
    rng = np.random.default_rng(seed)
    labels = np.memmap(path, dtype=np.uint16, mode="r")
    n = len(labels)

    counts = np.zeros(k, dtype=np.int64)
    reservoir = {cid: [] for cid in cluster_ids}
    seen = {cid: 0 for cid in cluster_ids}
    target_set = set(cluster_ids)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        chunk_data = np.array(labels[start:end])

        counts += np.bincount(chunk_data, minlength=k)

        for cid in target_set:
            matches = np.where(chunk_data == cid)[0] + start
            if len(matches) == 0:
                continue

            k0 = seen[cid]
            m = len(matches)
            seen[cid] = k0 + m

            if k0 < n_examples:
                fill = min(n_examples - k0, m)
                reservoir[cid].extend(matches[:fill].tolist())
                matches = matches[fill:]
                k0 += fill
                m -= fill

            if m > 0 and len(reservoir[cid]) == n_examples:
                ranks = np.arange(k0, k0 + m, dtype=np.int64)
                accept = rng.random(m) < (n_examples / (ranks + 1))
                if accept.any():
                    chosen = matches[accept]
                    slots = rng.integers(0, n_examples, size=len(chosen))
                    for gi, slot in zip(chosen.tolist(), slots.tolist()):
                        reservoir[cid][slot] = gi

    return counts, n, reservoir


def load_centroids(path):
    """Load centroid matrix, return None if unavailable."""
    if path and os.path.isfile(path):
        return np.load(path)
    return None


def try_import_hand_indexer():
    """Try to import the hand_indexer pybind module."""
    try:
        from hand_indexer import FlopIndexer

        return FlopIndexer()
    except ImportError as e:
        print(f"WARNING: {e} — example hands will be skipped.", file=sys.stderr)
        return None


# ── Statistics ───────────────────────────────────────────────────────
def gini(counts):
    """Gini coefficient of cluster sizes (0 = perfectly equal, 1 = maximally unequal)."""
    sorted_c = np.sort(counts)
    n = len(sorted_c)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_c) / (n * np.sum(sorted_c))) - (n + 1) / n


def print_stats(counts, n, k):
    used = np.count_nonzero(counts)
    print(f"\n{'=' * 60}")
    print(f"  Flop Label Statistics")
    print(f"{'=' * 60}")
    print(f"  Total labels:    {n:>16,}")
    print(f"  Clusters (K):    {k:>16,}")
    print(f"  Clusters used:   {used:>16,}  ({used / k * 100:.1f}%)")
    print(f"  Empty clusters:  {k - used:>16,}")
    print(f"{'-' * 60}")
    print(f"  Min size:        {counts.min():>16,}")
    print(f"  Max size:        {counts.max():>16,}")
    print(f"  Mean size:       {counts.mean():>16,.1f}")
    print(f"  Median size:     {int(np.median(counts)):>16,}")
    print(f"  Std dev:         {counts.std():>16,.1f}")
    print(f"  Gini coeff:      {gini(counts):>16.4f}")
    print(f"{'-' * 60}")

    top_idx = np.argsort(counts)[::-1]
    print("  Top 10 clusters:")
    for i in range(min(10, k)):
        ci = top_idx[i]
        print(f"    #{ci:<8d}  {counts[ci]:>12,} hands  ({counts[ci] / n * 100:.4f}%)")

    print("  Bottom 10 non-empty clusters:")
    bottom = np.argsort(counts)
    bottom_nz = bottom[counts[bottom] > 0][:10]
    for ci in bottom_nz:
        print(f"    #{ci:<8d}  {counts[ci]:>12,} hands  ({counts[ci] / n * 100:.6f}%)")
    print(f"{'=' * 60}\n")


# ── Plotting ─────────────────────────────────────────────────────────
def plot_cluster_sizes(axes, counts, n):
    """Three distribution plots: histogram, rank-size, CDF."""
    ax1, ax2, ax3 = axes

    ax1.hist(counts, bins=100, color="#2196F3", edgecolor="none", alpha=0.85)
    ax1.set_yscale("log")
    ax1.set_xlabel("Hands per cluster")
    ax1.set_ylabel("Number of clusters (log)")
    ax1.set_title("Cluster Size Distribution")
    ax1.axvline(
        counts.mean(),
        color="#F44336",
        ls="--",
        lw=1.2,
        label=f"mean={counts.mean():,.0f}",
    )
    ax1.axvline(
        np.median(counts),
        color="#FF9800",
        ls="--",
        lw=1.2,
        label=f"median={int(np.median(counts)):,}",
    )
    ax1.legend(fontsize=8)

    sorted_desc = np.sort(counts)[::-1]
    ranks = np.arange(1, len(sorted_desc) + 1)
    ax2.loglog(ranks, sorted_desc, color="#4CAF50", lw=0.8)
    ax2.set_xlabel("Cluster rank")
    ax2.set_ylabel("Cluster size (hands)")
    ax2.set_title("Rank–Size Plot (log-log)")
    ax2.grid(True, alpha=0.3, which="both")

    cumulative = np.cumsum(sorted_desc) / n
    ax3.plot(ranks / len(ranks) * 100, cumulative * 100, color="#9C27B0", lw=1.5)
    ax3.set_xlabel("Top N% of clusters")
    ax3.set_ylabel("% of hands covered")
    ax3.set_title("Cumulative Hand Coverage")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(50, color="gray", ls=":", lw=0.8)
    ax3.axvline(10, color="gray", ls=":", lw=0.8)
    idx50 = np.searchsorted(cumulative, 0.5)
    pct50 = idx50 / len(ranks) * 100
    ax3.annotate(
        f"50% at top {pct50:.1f}%",
        xy=(pct50, 50),
        fontsize=8,
        color="#9C27B0",
        xytext=(pct50 + 5, 40),
        arrowprops=dict(arrowstyle="->", color="#9C27B0"),
    )


def compute_centroid_features(centroids, expected_ehs):
    """Pre-compute shared features: E[EHS], PCA projection, variance explained."""
    mean_vec = centroids.mean(axis=0)
    centered = centroids - mean_vec
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt[:2].T  # (K, 2)
    var_explained = S[:2] ** 2 / np.sum(S**2) * 100
    return expected_ehs, proj, var_explained


# ── Card rendering ───────────────────────────────────────────────────
SUIT_SYMBOLS = {"c": "\u2663", "d": "\u2666", "h": "\u2665", "s": "\u2660"}
SUIT_COLORS = {"c": "#228B22", "d": "#1565C0", "h": "#CC0000", "s": "#222222"}
RANK_EXPAND = {"T": "10"}


def render_hand_line(ax, hand_str, x0, y, fontsize=9):
    """Render a hand like 'Ah Kd | Qs Jc 7h' with colored suit symbols."""
    x = x0
    cw = 0.015
    parts = hand_str.split("|")
    hole_cards = parts[0].strip().split() if len(parts) >= 1 else []
    board_cards = parts[1].strip().split() if len(parts) >= 2 else []

    for card in hole_cards:
        if len(card) < 2:
            continue
        rank_ch, suit_ch = card[:-1], card[-1]
        rank_display = RANK_EXPAND.get(rank_ch, rank_ch)
        suit_sym = SUIT_SYMBOLS.get(suit_ch, suit_ch)
        suit_col = SUIT_COLORS.get(suit_ch, "#333333")
        ax.text(
            x,
            y,
            rank_display,
            fontsize=fontsize,
            fontweight="bold",
            color="#333333",
            fontfamily="monospace",
            transform=ax.transAxes,
        )
        x += cw * len(rank_display)
        ax.text(
            x,
            y,
            suit_sym,
            fontsize=fontsize,
            fontweight="bold",
            color=suit_col,
            fontfamily="monospace",
            transform=ax.transAxes,
        )
        x += cw * 1.5

    ax.text(
        x,
        y,
        "|",
        fontsize=fontsize,
        color="#999999",
        fontfamily="monospace",
        transform=ax.transAxes,
    )
    x += cw * 2

    for card in board_cards:
        if len(card) < 2:
            continue
        rank_ch, suit_ch = card[:-1], card[-1]
        rank_display = RANK_EXPAND.get(rank_ch, rank_ch)
        suit_sym = SUIT_SYMBOLS.get(suit_ch, suit_ch)
        suit_col = SUIT_COLORS.get(suit_ch, "#333333")
        ax.text(
            x,
            y,
            rank_display,
            fontsize=fontsize,
            color="#555555",
            fontfamily="monospace",
            transform=ax.transAxes,
        )
        x += cw * len(rank_display)
        ax.text(
            x,
            y,
            suit_sym,
            fontsize=fontsize,
            color=suit_col,
            fontfamily="monospace",
            transform=ax.transAxes,
        )
        x += cw * 1.5


# ── Representatives figure ───────────────────────────────────────────
REP_PERCENTILES = [0, 1, 5, 20, 35, 50, 65, 80, 95, 99, 100]
REP_MARKERS = ["v", "v", "v", "s", "s", "o", "s", "s", "^", "^", "^"]
REP_CMAP = plt.cm.RdYlGn
REP_N_EXAMPLES = 5


def select_representatives(expected_ehs, counts):
    """Pick one cluster per EHS percentile. Returns indices array."""
    ehs_order = np.argsort(expected_ehs)
    k = len(expected_ehs)
    reps = []
    for pct in REP_PERCENTILES:
        pos = int(pct / 100 * (k - 1))
        reps.append(ehs_order[pos])
    return np.array(reps)


def plot_representatives(
    centroids, counts, expected_ehs, proj, var_explained, output_path
):
    """Figure 2: PCA and rank-size with representatives marked (1×2)."""
    reps = select_representatives(expected_ehs, counts)
    rep_colors = [REP_CMAP(p / 100) for p in REP_PERCENTILES]
    rep_labels = [
        f"P{p} (#{reps[i]}, E[EHS]={expected_ehs[reps[i]]:.2f})"
        for i, p in enumerate(REP_PERCENTILES)
    ]
    rank_of = np.argsort(np.argsort(counts)[::-1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        "Representative Cluster Deep-Dive", fontsize=14, fontweight="bold", y=0.98
    )

    ax1.scatter(
        proj[:, 0],
        proj[:, 1],
        c=expected_ehs,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        s=1.5,
        alpha=0.5,
        rasterized=True,
    )
    for i, ci in enumerate(reps):
        ax1.scatter(
            proj[ci, 0],
            proj[ci, 1],
            c=[rep_colors[i]],
            s=120,
            marker=REP_MARKERS[i],
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=rep_labels[i],
        )
    ax1.set_xlabel(f"PC1 ({var_explained[0]:.1f}% var)", fontsize=11)
    ax1.set_ylabel(f"PC2 ({var_explained[1]:.1f}% var)", fontsize=11)
    ax1.set_title("PCA of Flop Centroids (color = E[EHS])", fontsize=12)
    ax1.legend(fontsize=7.5, loc="upper left", markerscale=0.8, ncol=2)

    sorted_desc = np.sort(counts)[::-1]
    ranks_arr = np.arange(1, len(sorted_desc) + 1)
    ax2.loglog(ranks_arr, sorted_desc, color="#BBBBBB", lw=0.8, zorder=1)
    for i, ci in enumerate(reps):
        r = rank_of[ci] + 1
        ax2.scatter(
            r,
            counts[ci],
            c=[rep_colors[i]],
            s=100,
            marker=REP_MARKERS[i],
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=f"#{ci}: {counts[ci]:,} hands (rank {r:,})",
        )
    ax2.set_xlabel("Cluster rank", fontsize=11)
    ax2.set_ylabel("Cluster size (hands)", fontsize=11)
    ax2.set_title("Representatives on Rank-Size Curve", fontsize=12)
    ax2.legend(fontsize=7, loc="upper right", ncol=2)
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Representatives figure saved to {output_path}")
    plt.close(fig)


def plot_hands(counts, expected_ehs, output_path, hand_examples):
    """Figure 3: example hands for all representatives in two columns."""
    reps = select_representatives(expected_ehs, counts)
    rep_colors = [REP_CMAP(p / 100) for p in REP_PERCENTILES]
    rank_of = np.argsort(np.argsort(counts)[::-1])
    n_reps = len(reps)

    n_left = n_reps // 2
    n_right = n_reps - n_left

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 13))
    fig.subplots_adjust(wspace=0.05)
    fig.suptitle(
        "Example Hands by E[EHS] Percentile (hole | flop board)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for ax_col, start_i, count in [(ax_left, 0, n_left), (ax_right, n_left, n_right)]:
        ax_col.axis("off")
        y = 0.96
        for j in range(count):
            i = start_i + j
            ci = reps[i]
            color = rep_colors[i]
            pct = REP_PERCENTILES[i]
            ehs = expected_ehs[ci]
            size = counts[ci]

            ax_col.text(
                0.0,
                y,
                f"P{pct}",
                fontsize=11,
                fontweight="bold",
                color=color,
                fontfamily="monospace",
                transform=ax_col.transAxes,
            )
            ax_col.text(
                0.08,
                y,
                f"E[EHS]={ehs:.2f}   {size:,} hands   (rank {rank_of[ci] + 1:,})",
                fontsize=10,
                color="#333333",
                fontfamily="monospace",
                transform=ax_col.transAxes,
            )
            y -= 0.030

            hands = hand_examples.get(ci, [])
            for h in hands:
                render_hand_line(ax_col, h, x0=0.08, y=y, fontsize=10)
                y -= 0.026
            y -= 0.018

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Hands figure saved to {output_path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    labels_path = os.path.abspath(LABELS_PATH)
    centroids_path = os.path.abspath(CENTROIDS_PATH)
    ehs_path = os.path.abspath(EHS_PATH)
    output_path = os.path.abspath(OUTPUT_PATH)
    k = K

    if not os.path.isfile(labels_path):
        sys.exit(f"Error: labels file not found: {labels_path}")

    # Load flop centroids
    centroids = load_centroids(centroids_path)
    has_centroids = centroids is not None
    if has_centroids:
        print(f"Flop centroids loaded: {centroids.shape}")
    else:
        print(f"No flop centroids found at {centroids_path} — skipping centroid plots.")

    # Load E[EHS] per cluster from file.
    expected_ehs = proj = var_explained = None
    if has_centroids:
        if not os.path.isfile(ehs_path):
            print(
                f"WARNING: EHS file not found at {ehs_path} — skipping centroid plots."
            )
            has_centroids = False
        else:
            print(f"EHS loaded from {ehs_path}")
            expected_ehs, proj, var_explained = compute_centroid_features(
                centroids, np.fromfile(ehs_path, dtype=np.float32)
            )

    # Single-pass scan: counts + reservoir samples for representative clusters
    indexer = try_import_hand_indexer()
    hand_examples = None

    if has_centroids and indexer:
        reps = select_representatives(expected_ehs, np.ones(k, dtype=np.int64))
        print(f"Loading labels from {labels_path} (single pass: counts + examples) ...")
        counts, n, idx_map = load_counts_and_examples(
            labels_path, k, reps.tolist(), n_examples=REP_N_EXAMPLES
        )
        print(f"  {n:,} labels loaded, K={k}")

        all_indices, index_owners = [], []
        for ci in reps:
            for idx in idx_map[ci]:
                index_owners.append(ci)
                all_indices.append(idx)
        if all_indices:
            card_lines = indexer.batch_unindex(all_indices)
            hand_examples = {ci: [] for ci in reps}
            for owner, line in zip(index_owners, card_lines):
                hand_examples[owner].append(line)
            print(f"  {len(card_lines)} example hands resolved.")
    else:
        print(f"Loading labels from {labels_path} ...")
        counts, n = load_label_counts(labels_path, k)
        print(f"  {n:,} labels loaded, K={k}")

    print_stats(counts, n, k)

    # ── Figure 1: Distribution overview (1×3) ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Flop Bucket Label Analysis — {n:,} hands  ×  {k:,} clusters",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plot_cluster_sizes(axes, counts, n)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Overview figure saved to {output_path}")
    plt.close(fig)

    # ── Figure 2: PCA + rank-size with representatives ───────────────
    if has_centroids:
        base, ext = os.path.splitext(output_path)
        reps_path = base + "_representatives" + ext
        plot_representatives(
            centroids, counts, expected_ehs, proj, var_explained, reps_path
        )

    # ── Figure 3: Example hands ───────────────────────────────────────
    if has_centroids:
        hands_path = base + "_hands" + ext
        if hand_examples is not None:
            plot_hands(counts, expected_ehs, hands_path, hand_examples)
        else:
            print("Skipping example hands (FlopIndexer not available).")


if __name__ == "__main__":
    main()
