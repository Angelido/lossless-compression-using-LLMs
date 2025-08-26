#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot throughput vs compressed percentage for each language (one point per compression algorithm).

Data requirements (CSV: Plain_Compression_Info_Final.csv):
- original_size_bytes      : original uncompressed size in bytes
- compressed_size_bytes    : compressed size in bytes
- throughput_MBps          : numeric throughput in MB/s
- language                 : language identifier
- compression              : compression algorithm name

Behavior:
- X axis: compressed percentage = 100 * compressed_size_bytes / original_size_bytes
- Y axis: throughput_MBps
- One subplot per language
- Each point is one compression algorithm (values averaged over rows per language+compression)
- The compression name is drawn next to each point (no legend)
- Points are color-coded by compression algorithm (consistent across subplots)
- The figure is always shown; saving is optional via --save

Usage:
    python plotPlainCompressionResults.py \
        --csv Plain_Compression_Info_Final.csv \
        --outdir ./out \
        --grid-cols 3 \
        --dpi 140 \
        [--save]

Notes for maintainers:
- Comments are in English per the user's preference.
"""

import argparse
import math
import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def unique_in_order(series: pd.Series) -> list:
    """Return unique values preserving their first-seen order."""
    seen, out = set(), []
    for x in series:
        if pd.isna(x):
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Compute compressed percentage and aggregate to one row per (language, compression)."""
    # Ensure numeric types
    df = df.copy()
    df["original_size_bytes"] = pd.to_numeric(df["original_size_bytes"], errors="coerce")
    df["compressed_size_bytes"] = pd.to_numeric(df["compressed_size_bytes"], errors="coerce")
    df["throughput_MBps"] = pd.to_numeric(df["throughput_MBps"], errors="coerce")

    # Compute compressed percentage (0..100)
    df["compressed_percentage"] = 100.0 * (df["compressed_size_bytes"] / df["original_size_bytes"].replace(0, np.nan))

    # Drop rows with missing essentials
    df = df.dropna(subset=["compressed_percentage", "throughput_MBps", "language", "compression"])

    # Aggregate: one point per (language, compression). Using mean for both axes.
    grouped = (
        df.groupby(["language", "compression"], as_index=False)
          .agg(compressed_percentage=("compressed_percentage", "mean"),
               throughput_MBps=("throughput_MBps", "mean"))
    )
    return grouped


def plot_by_language(grouped: pd.DataFrame, outdir: str, grid_cols: int, dpi: int, save_flag: bool) -> str:
    """Plot multi-panel scatter with text labels, show figure, optionally save."""
    os.makedirs(outdir, exist_ok=True)

    languages = unique_in_order(grouped["language"])
    if not languages:
        raise ValueError("No 'language' values found in the CSV.")

    # Build a consistent color map across all subplots
    compressions = unique_in_order(grouped["compression"])
    cmap = plt.get_cmap("tab20")
    color_map = {comp: cmap(i % 20) for i, comp in enumerate(compressions)}

    n = len(languages)
    cols = max(1, grid_cols)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.8 * rows), squeeze=False)
    axes = axes.ravel()

    # Plot each language
    for idx, lang in enumerate(languages):
        ax = axes[idx]
        sub = grouped[grouped["language"] == lang]

        # Scatter points per compression for color consistency
        for comp in compressions:
            data = sub[sub["compression"] == comp]
            if data.empty:
                continue
            ax.scatter(
                data["compressed_percentage"],
                data["throughput_MBps"],
                s=50,
                alpha=0.9,
                c=[color_map[comp]]
            )
            # Add compression labels near each point
            for _, r in data.iterrows():
                ax.annotate(
                    str(r["compression"]),
                    (r["compressed_percentage"], r["throughput_MBps"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8
                )

        ax.set_title(str(lang))
        ax.set_xlabel("Compressed percentage (%)")
        ax.set_ylabel("Throughput (MB/s)")
        ax.grid(True, linestyle="--", alpha=0.4)

    # Remove unused axes, if any
    for j in range(idx + 1, rows * cols):
        fig.delaxes(axes[j])

    # English main title
    fig.suptitle("Throughput vs Compressed Percentage by Language", y=1.02, fontsize=14)
    fig.tight_layout()

    outpath = os.path.join(outdir, "throughput_vs_compressed_percentage_by_language.png")
    if save_flag:
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")

    # Always show the figure
    plt.show()
    plt.close(fig)

    return outpath if save_flag else ""


def main():
    parser = argparse.ArgumentParser(description="Plot throughput vs compressed percentage by language.")
    parser.add_argument("--csv", default="Results/Plain_Compression_Info_Final.csv",
                        help="Path to the input CSV (default: Plain_Compression_Info_Final.csv)")
    parser.add_argument("--outdir", default="./out", help="Output directory for the figure (default: ./out)")
    parser.add_argument("--grid-cols", type=int, default=3, help="Number of subplot columns (default: 3)")
    parser.add_argument("--dpi", type=int, default=140, help="Figure DPI if saving (default: 140)")
    parser.add_argument("--save", action="store_true", help="Save the figure to --outdir (shown regardless).")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    grouped = prepare_data(df)
    saved_path = plot_by_language(grouped, args.outdir, args.grid_cols, args.dpi, args.save)

    if args.save and saved_path:
        print(f"Saved: {saved_path}")


if __name__ == "__main__":
    main()
