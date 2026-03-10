#!/usr/bin/env python3
"""Visualize river bucket label distribution and centroid structure.

Reads the binary river_labels.bin (2.4B uint16 cluster labels) and
optionally river_centroids.npy (K×169 float32) to produce a 2×3
diagnostic figure plus console summary statistics.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ── CLI ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", default=os.path.join(
        os.path.dirname(__file__), "..", "output", "river_labels.bin"),
        help="Path to river_labels.bin (uint16 binary)")
    p.add_argument("--centroids", default=None,
        help="Path to river_centroids.npy (optional, auto-detected)")
    p.add_argument("-k", type=int, default=30_000,
        help="Number of clusters (default: 30000)")
    p.add_argument("-o", "--output", default=None,
        help="Output PNG path (default: <labels_dir>/river_labels_viz.png)")
    return p.parse_args()


# ── Data loading ─────────────────────────────────────────────────────
def load_label_counts(path, k, chunk=100_000_000):
    """Memory-map labels and compute per-cluster counts in chunks."""
    labels = np.memmap(path, dtype=np.uint16, mode="r")
    n = len(labels)
    counts = np.zeros(k, dtype=np.int64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        counts += np.bincount(labels[start:end], minlength=k)
    return counts, n


def load_centroids(path):
    """Load centroid matrix, return None if unavailable."""
    if path and os.path.isfile(path):
        return np.load(path)
    return None


def try_import_hand_indexer():
    """Try to import the hand_indexer pybind module."""
    try:
        from hand_indexer import RiverIndexer
        return RiverIndexer()
    except ImportError:
        return None


def find_example_indices(labels_path, cluster_ids, n_examples=5, chunk=100_000_000):
    """Scan labels memmap to find example hand indices for each cluster."""
    labels = np.memmap(labels_path, dtype=np.uint16, mode="r")
    n = len(labels)
    # {cluster_id: [list of hand indices]}
    result = {cid: [] for cid in cluster_ids}
    targets = set(cluster_ids)
    done = set()

    for start in range(0, n, chunk):
        if len(done) == len(targets):
            break
        end = min(start + chunk, n)
        chunk_data = np.array(labels[start:end])  # copy into RAM for fast ops
        for cid in targets - done:
            matches = np.where(chunk_data == cid)[0]
            needed = n_examples - len(result[cid])
            if needed > 0 and len(matches) > 0:
                # Pick evenly spaced examples from matches
                if len(matches) <= needed:
                    pick = matches
                else:
                    pick = matches[np.linspace(0, len(matches)-1, needed, dtype=int)]
                result[cid].extend((start + pick).tolist())
                if len(result[cid]) >= n_examples:
                    done.add(cid)
    return result


# ── Statistics ───────────────────────────────────────────────────────
def gini(counts):
    """Gini coefficient of cluster sizes (0 = perfectly equal, 1 = maximally unequal)."""
    sorted_c = np.sort(counts)
    n = len(sorted_c)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_c) / (n * np.sum(sorted_c))) - (n + 1) / n


def print_stats(counts, n, k):
    used = np.count_nonzero(counts)
    print(f"\n{'='*60}")
    print(f"  River Label Statistics")
    print(f"{'='*60}")
    print(f"  Total labels:    {n:>16,}")
    print(f"  Clusters (K):    {k:>16,}")
    print(f"  Clusters used:   {used:>16,}  ({used/k*100:.1f}%)")
    print(f"  Empty clusters:  {k - used:>16,}")
    print(f"{'-'*60}")
    print(f"  Min size:        {counts.min():>16,}")
    print(f"  Max size:        {counts.max():>16,}")
    print(f"  Mean size:       {counts.mean():>16,.1f}")
    print(f"  Median size:     {int(np.median(counts)):>16,}")
    print(f"  Std dev:         {counts.std():>16,.1f}")
    print(f"  Gini coeff:      {gini(counts):>16.4f}")
    print(f"{'-'*60}")

    top_idx = np.argsort(counts)[::-1]
    print("  Top 10 clusters:")
    for i in range(min(10, k)):
        ci = top_idx[i]
        print(f"    #{ci:<8d}  {counts[ci]:>12,} hands  ({counts[ci]/n*100:.4f}%)")

    print("  Bottom 10 non-empty clusters:")
    nonzero = top_idx[top_idx < used] if used < k else top_idx
    bottom = np.argsort(counts)
    bottom_nz = bottom[counts[bottom] > 0][:10]
    for ci in bottom_nz:
        print(f"    #{ci:<8d}  {counts[ci]:>12,} hands  ({counts[ci]/n*100:.6f}%)")
    print(f"{'='*60}\n")


# ── Plotting ─────────────────────────────────────────────────────────
def plot_cluster_sizes(axes, counts, n):
    """Top row: histogram, rank-size, CDF."""
    ax1, ax2, ax3 = axes

    # 1) Histogram of cluster sizes (log Y to reveal the tail)
    ax1.hist(counts, bins=200, color="#2196F3", edgecolor="none", alpha=0.85)
    ax1.set_yscale("log")
    ax1.set_xlabel("Hands per cluster")
    ax1.set_ylabel("Number of clusters (log)")
    ax1.set_title("Cluster Size Distribution")
    ax1.axvline(counts.mean(), color="#F44336", ls="--", lw=1.2, label=f"mean={counts.mean():,.0f}")
    ax1.axvline(np.median(counts), color="#FF9800", ls="--", lw=1.2, label=f"median={int(np.median(counts)):,}")
    ax1.legend(fontsize=8)

    # 2) Rank-size (log-log)
    sorted_desc = np.sort(counts)[::-1]
    ranks = np.arange(1, len(sorted_desc) + 1)
    ax2.loglog(ranks, sorted_desc, color="#4CAF50", lw=0.8)
    ax2.set_xlabel("Cluster rank")
    ax2.set_ylabel("Cluster size (hands)")
    ax2.set_title("Rank–Size Plot (log-log)")
    ax2.grid(True, alpha=0.3, which="both")

    # 3) CDF — fraction of hands covered by top N clusters
    cumulative = np.cumsum(sorted_desc) / n
    ax3.plot(ranks / len(ranks) * 100, cumulative * 100, color="#9C27B0", lw=1.5)
    ax3.set_xlabel("Top N% of clusters")
    ax3.set_ylabel("% of hands covered")
    ax3.set_title("Cumulative Hand Coverage")
    ax3.grid(True, alpha=0.3)
    # Mark 10% and 50% lines
    ax3.axhline(50, color="gray", ls=":", lw=0.8)
    ax3.axvline(10, color="gray", ls=":", lw=0.8)
    # Find and annotate where 50% coverage occurs
    idx50 = np.searchsorted(cumulative, 0.5)
    pct50 = idx50 / len(ranks) * 100
    ax3.annotate(f"50% at top {pct50:.1f}%", xy=(pct50, 50),
                 fontsize=8, color="#9C27B0",
                 xytext=(pct50 + 10, 40), arrowprops=dict(arrowstyle="->", color="#9C27B0"))


def compute_centroid_features(centroids):
    """Pre-compute shared features: mean EHS, PCA projection, variance explained."""
    mean_ehs = centroids.mean(axis=1) / 255.0  # true equity 0.0-1.0
    mean_vec = centroids.mean(axis=0)
    centered = centroids - mean_vec
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt[:2].T  # (K, 2)
    var_explained = S[:2] ** 2 / np.sum(S ** 2) * 100
    return mean_ehs, proj, var_explained


def plot_centroids(axes, centroids, counts, mean_ehs, proj, var_explained):
    """Bottom row: PCA scatter, equity profiles, similarity heatmap."""
    ax4, ax5, ax6 = axes
    k = centroids.shape[0]

    # 4) PCA projection colored by mean EHS
    sc = ax4.scatter(proj[:, 0], proj[:, 1],
                     c=mean_ehs, cmap="RdYlGn", vmin=0, vmax=1,
                     s=1, alpha=0.5, rasterized=True)
    ax4.set_xlabel(f"PC1 ({var_explained[0]:.1f}% var)")
    ax4.set_ylabel(f"PC2 ({var_explained[1]:.1f}% var)")
    ax4.set_title("PCA of Centroids (color = mean EHS)")
    plt.colorbar(sc, ax=ax4, label="Mean EHS (equity)", shrink=0.8)

    # 5) Mean equity profile by cluster-size quintile (true equity 0.0-1.0)
    quintile_labels = ["Smallest 20%", "Q2", "Q3", "Q4", "Largest 20%"]
    size_order = np.argsort(counts)
    quintile_size = k // 5
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, 5))
    x_buckets = np.arange(centroids.shape[1])
    for qi in range(5):
        start = qi * quintile_size
        end = (qi + 1) * quintile_size if qi < 4 else k
        idx = size_order[start:end]
        mean_profile = centroids[idx].mean(axis=0) / 255.0
        ax5.plot(x_buckets, mean_profile, color=colors[qi],
                 lw=1.2, label=quintile_labels[qi], alpha=0.9)
    # Preflop category boundaries: Pairs 0-12 | Suited 13-90 | Offsuit 91-168
    ax5.axvline(13, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax5.axvline(91, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax5.text(6, 0.02, "Pairs", ha="center", fontsize=7, color="gray")
    ax5.text(52, 0.02, "Suited", ha="center", fontsize=7, color="gray")
    ax5.text(130, 0.02, "Offsuit", ha="center", fontsize=7, color="gray")
    ax5.set_xlabel("Opponent preflop bucket (0-168)")
    ax5.set_ylabel("Equity")
    ax5.set_ylim(0, 1.0)
    ax5.set_title("Equity Profile by Cluster Size Quintile")
    ax5.legend(fontsize=7, loc="upper left")
    ax5.grid(True, alpha=0.2)

    # 6) Cosine similarity heatmap sorted by mean EHS
    sample_n = min(200, k)
    ehs_order = np.argsort(mean_ehs)
    sample_idx = np.linspace(0, k - 1, sample_n, dtype=int)
    sample_idx = ehs_order[sample_idx]
    sample_c = centroids[sample_idx]
    norms = np.linalg.norm(sample_c, axis=1, keepdims=True).clip(1e-10)
    normed = sample_c / norms
    sim = normed @ normed.T
    im = ax6.imshow(sim, cmap="RdBu_r", vmin=-0.2, vmax=1.0,
                    aspect="auto", interpolation="nearest")
    ax6.set_xlabel("Centroid (sorted by mean EHS)")
    ax6.set_ylabel("Centroid (sorted by mean EHS)")
    ax6.set_title("Cosine Similarity (200 sampled)")
    plt.colorbar(im, ax=ax6, shrink=0.8)


# ── Card rendering ───────────────────────────────────────────────────
SUIT_SYMBOLS = {"c": "\u2663", "d": "\u2666", "h": "\u2665", "s": "\u2660"}
SUIT_COLORS  = {"c": "#228B22", "d": "#1565C0", "h": "#CC0000", "s": "#222222"}
RANK_EXPAND  = {"T": "10"}


def render_hand_line(ax, hand_str, x0, y, fontsize=9):
    """Render a hand like 'Ah Kd | Qs Jc 7h 2s 9d' with colored suit symbols."""
    x = x0
    # Character width in axes fraction (approximate for monospace at this size)
    cw = 0.015
    parts = hand_str.split("|")
    hole_cards = parts[0].strip().split() if len(parts) >= 1 else []
    board_cards = parts[1].strip().split() if len(parts) >= 2 else []

    for ci, card in enumerate(hole_cards):
        if len(card) < 2:
            continue
        rank_ch, suit_ch = card[:-1], card[-1]
        rank_display = RANK_EXPAND.get(rank_ch, rank_ch)
        suit_sym = SUIT_SYMBOLS.get(suit_ch, suit_ch)
        suit_col = SUIT_COLORS.get(suit_ch, "#333333")
        # Render rank in dark, suit in color
        ax.text(x, y, rank_display, fontsize=fontsize, fontweight="bold",
                color="#333333", fontfamily="monospace", transform=ax.transAxes)
        x += cw * len(rank_display)
        ax.text(x, y, suit_sym, fontsize=fontsize, fontweight="bold",
                color=suit_col, fontfamily="monospace", transform=ax.transAxes)
        x += cw * 1.5  # gap after card

    # Separator
    ax.text(x, y, "|", fontsize=fontsize, color="#999999",
            fontfamily="monospace", transform=ax.transAxes)
    x += cw * 2

    for card in board_cards:
        if len(card) < 2:
            continue
        rank_ch, suit_ch = card[:-1], card[-1]
        rank_display = RANK_EXPAND.get(rank_ch, rank_ch)
        suit_sym = SUIT_SYMBOLS.get(suit_ch, suit_ch)
        suit_col = SUIT_COLORS.get(suit_ch, "#333333")
        ax.text(x, y, rank_display, fontsize=fontsize,
                color="#555555", fontfamily="monospace", transform=ax.transAxes)
        x += cw * len(rank_display)
        ax.text(x, y, suit_sym, fontsize=fontsize,
                color=suit_col, fontfamily="monospace", transform=ax.transAxes)
        x += cw * 1.5


# ── Representatives figure ───────────────────────────────────────────
REP_PERCENTILES = [0, 1, 5, 20, 35, 50, 65, 80, 95, 99, 100]
REP_MARKERS = ["v", "v", "v", "s", "s", "o", "s", "s", "^", "^", "^"]
REP_CMAP = plt.cm.RdYlGn
REP_N_EXAMPLES = 5


def select_representatives(mean_ehs, counts):
    """Pick one cluster per EHS percentile. Returns (indices, labels)."""
    ehs_order = np.argsort(mean_ehs)
    k = len(mean_ehs)
    reps = []
    for pct in REP_PERCENTILES:
        pos = int(pct / 100 * (k - 1))
        reps.append(ehs_order[pos])
    return np.array(reps)


def plot_representatives(centroids, counts, mean_ehs, proj, var_explained,
                         output_path):
    """Second figure: PCA and rank-size with representatives marked (1×2)."""
    reps = select_representatives(mean_ehs, counts)
    rep_colors = [REP_CMAP(p / 100) for p in REP_PERCENTILES]
    rep_labels = [f"P{p} (#{reps[i]}, EHS={mean_ehs[reps[i]]:.2f})"
                  for i, p in enumerate(REP_PERCENTILES)]
    rank_of = np.argsort(np.argsort(counts)[::-1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Representative Cluster Deep-Dive", fontsize=14,
                 fontweight="bold", y=0.98)

    # Left: PCA with representatives highlighted
    ax1.scatter(proj[:, 0], proj[:, 1], c="lightgray", s=0.3, alpha=0.3,
                rasterized=True)
    for i, ci in enumerate(reps):
        ax1.scatter(proj[ci, 0], proj[ci, 1],
                    c=[rep_colors[i]], s=120, marker=REP_MARKERS[i],
                    edgecolors="black", linewidths=0.8, zorder=5,
                    label=rep_labels[i])
    ax1.set_xlabel(f"PC1 ({var_explained[0]:.1f}% var)", fontsize=11)
    ax1.set_ylabel(f"PC2 ({var_explained[1]:.1f}% var)", fontsize=11)
    ax1.set_title("Representatives on PCA Map", fontsize=12)
    ax1.legend(fontsize=7.5, loc="upper left", markerscale=0.8, ncol=2)

    # Right: Rank-size with representatives marked
    sorted_desc = np.sort(counts)[::-1]
    ranks_arr = np.arange(1, len(sorted_desc) + 1)
    ax2.loglog(ranks_arr, sorted_desc, color="#BBBBBB", lw=0.8, zorder=1)
    for i, ci in enumerate(reps):
        r = rank_of[ci] + 1
        ax2.scatter(r, counts[ci], c=[rep_colors[i]], s=100,
                    marker=REP_MARKERS[i], edgecolors="black", linewidths=0.8,
                    zorder=5, label=f"#{ci}: {counts[ci]:,} hands (rank {r:,})")
    ax2.set_xlabel("Cluster rank", fontsize=11)
    ax2.set_ylabel("Cluster size (hands)", fontsize=11)
    ax2.set_title("Representatives on Rank-Size Curve", fontsize=12)
    ax2.legend(fontsize=7, loc="upper right", ncol=2)
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Representatives figure saved to {output_path}")
    plt.close(fig)


def plot_hands(counts, mean_ehs, output_path, hand_examples):
    """Third figure: example hands for all representatives in two columns."""
    reps = select_representatives(mean_ehs, counts)
    rep_colors = [REP_CMAP(p / 100) for p in REP_PERCENTILES]
    rank_of = np.argsort(np.argsort(counts)[::-1])
    n_reps = len(reps)

    # Split into two columns: first 5 left, remaining 6 right
    n_left = n_reps // 2
    n_right = n_reps - n_left

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 13))
    fig.subplots_adjust(wspace=0.05)
    fig.suptitle("Example Hands by EHS Percentile (hole | board)",
                 fontsize=14, fontweight="bold", y=0.98)

    for ax_col, start_i, count in [(ax_left, 0, n_left),
                                    (ax_right, n_left, n_right)]:
        ax_col.axis("off")
        y = 0.96
        for j in range(count):
            i = start_i + j
            ci = reps[i]
            color = rep_colors[i]
            pct = REP_PERCENTILES[i]
            ehs = mean_ehs[ci]
            size = counts[ci]

            # Header with colored percentile badge
            ax_col.text(0.0, y, f"P{pct}",
                        fontsize=11, fontweight="bold", color=color,
                        fontfamily="monospace", transform=ax_col.transAxes)
            ax_col.text(0.08, y,
                        f"EHS={ehs:.2f}   {size:,} hands   (rank {rank_of[ci]+1:,})",
                        fontsize=10, color="#333333",
                        fontfamily="monospace", transform=ax_col.transAxes)
            y -= 0.030

            # Example hands
            hands = hand_examples.get(ci, [])
            for h in hands:
                render_hand_line(ax_col, h, x0=0.08, y=y, fontsize=10)
                y -= 0.026
            y -= 0.018  # gap between reps

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Hands figure saved to {output_path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Resolve paths
    labels_path = os.path.abspath(args.labels)
    if not os.path.isfile(labels_path):
        sys.exit(f"Error: labels file not found: {labels_path}")

    centroids_path = args.centroids
    if centroids_path is None:
        # Auto-detect next to labels
        centroids_path = os.path.join(os.path.dirname(labels_path), "river_centroids.npy")

    output_path = args.output or os.path.join(
        os.path.dirname(labels_path), "river_labels_viz.png")

    k = args.k

    # Load data
    print(f"Loading labels from {labels_path} ...")
    counts, n = load_label_counts(labels_path, k)
    print(f"  {n:,} labels loaded, K={k}")

    centroids = load_centroids(centroids_path)
    has_centroids = centroids is not None
    if has_centroids:
        print(f"Centroids loaded: {centroids.shape}")
    else:
        print(f"No centroids file found at {centroids_path} — skipping equity plots.")

    # Print console stats
    print_stats(counts, n, k)

    # Pre-compute shared centroid features
    mean_ehs = proj = var_explained = None
    if has_centroids:
        mean_ehs, proj, var_explained = compute_centroid_features(centroids)

    # ── Figure 1: Overview (2×3) ─────────────────────────────────────
    nrows = 2 if has_centroids else 1
    fig, axes = plt.subplots(nrows, 3, figsize=(18, 6 * nrows))
    fig.suptitle(
        f"River Bucket Label Analysis — {n:,} hands  ×  {k:,} clusters",
        fontsize=14, fontweight="bold", y=0.98)

    top_axes = axes[0] if nrows == 2 else axes
    plot_cluster_sizes(top_axes, counts, n)

    if has_centroids:
        plot_centroids(axes[1], centroids, counts, mean_ehs, proj, var_explained)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Overview figure saved to {output_path}")
    plt.close(fig)

    # ── Figure 2: Representative clusters on PCA + rank-size ────────
    if has_centroids:
        base, ext = os.path.splitext(output_path)
        reps_path = base + "_representatives" + ext
        plot_representatives(centroids, counts, mean_ehs, proj,
                             var_explained, reps_path)

    # ── Figure 3: Example hands ───────────────────────────────────
    if has_centroids:
        hands_path = base + "_hands" + ext
        indexer = try_import_hand_indexer()
        if indexer:
            reps = select_representatives(mean_ehs, counts)
            print(f"Finding example hands (hand_indexer module) ...")
            idx_map = find_example_indices(labels_path, reps.tolist(),
                                           n_examples=REP_N_EXAMPLES)
            all_indices = []
            index_owners = []
            for ci in reps:
                for idx in idx_map[ci]:
                    index_owners.append(ci)
                    all_indices.append(idx)
            if all_indices:
                card_lines = indexer.batch_unindex(all_indices)
                hand_examples = {ci: [] for ci in reps}
                for owner, line in zip(index_owners, card_lines):
                    hand_examples[owner].append(line)
                print(f"  {len(card_lines)} hands resolved.")
                plot_hands(counts, mean_ehs, hands_path, hand_examples)
        else:
            print("hand_indexer module not found -- run 'make pybind' first.")


if __name__ == "__main__":
    main()
