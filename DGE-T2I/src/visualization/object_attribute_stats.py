#!/usr/bin/env python3
"""Object-level attribute pairing statistics using spaCy dependencies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.pos_stats import (  # noqa: E402
    BENCHMARK_COLORS,
    DEFAULT_OURS,
    ROOT,
    load_benchmarks,
    load_nlp,
)

DEFAULT_OUTPUT_DIR = ROOT / "DGE-T2I/reports/visualization/object_attribute_stats"
ENTITY_POS = {"NOUN", "PROPN"}
ATTRIBUTE_DEPS = {"amod", "acomp", "advmod"}


def paired_attributes(entity) -> list[str]:
    attrs = [child.lemma_.lower() for child in entity.children if child.pos_ == "ADJ" and child.dep_ in ATTRIBUTE_DEPS]

    # Capture common attribute phrase patterns such as "person with blonde hair"
    # by attaching adjectival modifiers of prepositional-object nouns back to the source entity.
    for child in entity.children:
        if child.dep_ == "prep":
            for grandchild in child.children:
                if grandchild.pos_ in ENTITY_POS:
                    attrs.extend(
                        attr.lemma_.lower()
                        for attr in grandchild.children
                        if attr.pos_ == "ADJ" and attr.dep_ in ATTRIBUTE_DEPS
                    )
    return attrs


def analyze_dataset(name: str, prompts: list[str], nlp) -> pd.DataFrame:
    rows = []
    for prompt_index, doc in enumerate(nlp.pipe(prompts, batch_size=256)):
        object_index = 0
        for token in doc:
            if token.pos_ not in ENTITY_POS:
                continue
            attrs = paired_attributes(token)
            rows.append(
                {
                    "dataset": name,
                    "prompt_index": prompt_index,
                    "object_index": object_index,
                    "object": token.lemma_.lower(),
                    "attribute_count": len(attrs),
                    "attributes": "; ".join(attrs),
                    "prompt": prompts[prompt_index],
                }
            )
            object_index += 1
    return pd.DataFrame(rows)


def write_summary(object_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary = (
        object_df.groupby("dataset")
        .agg(
            Objects=("attribute_count", "size"),
            **{
                "Avg Attributes/Object": ("attribute_count", "mean"),
                "Median Attributes/Object": ("attribute_count", "median"),
                "Pct Objects w/Attributes": ("attribute_count", lambda s: (s > 0).mean() * 100),
                "Max Attributes/Object": ("attribute_count", "max"),
            },
        )
        .reset_index()
        .rename(columns={"dataset": "Dataset"})
        .sort_values("Avg Attributes/Object", ascending=False)
    )
    summary.to_csv(output_dir / "object_attribute_summary.csv", index=False)
    object_df.to_csv(output_dir / "object_attribute_counts.csv", index=False)
    return summary


def style_axis(ax, title: str, ylabel: str = "") -> None:
    ax.set_title(title, fontsize=18, fontweight="semibold", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="x", rotation=30, labelsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_object_attribute_boxplot(object_df: pd.DataFrame, output_dir: Path) -> None:
    stats = object_df.groupby("dataset")["attribute_count"].agg(["mean", "median"]).sort_values(
        ["mean", "median"], ascending=False
    )
    labels = stats.index.tolist()
    data = [object_df.loc[object_df["dataset"] == label, "attribute_count"].to_numpy() for label in labels]

    fig, ax = plt.subplots(figsize=(15, 8))
    box = ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
        widths=0.62,
        medianprops={"color": "#f28e2b", "linewidth": 2.2},
        boxprops={"linewidth": 1.3, "edgecolor": "#333333"},
        whiskerprops={"linewidth": 1.2, "color": "#333333"},
        capprops={"linewidth": 1.2, "color": "#333333"},
    )
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(BENCHMARK_COLORS.get(label, "#cccccc"))
        patch.set_alpha(0.82 if label != "Ours" else 0.96)
        if label == "Ours":
            patch.set_linewidth(2.4)

    means = [values.mean() if len(values) else 0.0 for values in data]
    ax.scatter(np.arange(1, len(labels) + 1), means, marker="D", s=58, color="#1b9e77", edgecolor="white", linewidth=0.8, zorder=3, label="Mean")
    style_axis(ax, "Object-Level Paired Attributes", "Paired attributes per object")
    ax.set_xlabel("Dataset (ordered by mean object-level paired attributes)", fontsize=14)
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "object_attribute_count_boxplot_ranked.png", dpi=240)
    plt.close(fig)


def plot_mean_bar(summary: pd.DataFrame, output_dir: Path) -> None:
    ranked = summary.sort_values("Avg Attributes/Object", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    ax.barh(
        ranked["Dataset"],
        ranked["Avg Attributes/Object"],
        color=[BENCHMARK_COLORS.get(name, "#cccccc") for name in ranked["Dataset"]],
        edgecolor=["#111111" if name == "Ours" else "white" for name in ranked["Dataset"]],
        linewidth=[2.2 if name == "Ours" else 0.8 for name in ranked["Dataset"]],
        alpha=0.92,
    )
    xmax = ranked["Avg Attributes/Object"].max()
    for y, value in enumerate(ranked["Avg Attributes/Object"]):
        ax.text(value + xmax * 0.015, y, f"{value:.2f}", va="center", fontsize=11)
    ax.set_title("Mean Paired Attributes per Object", fontsize=20, fontweight="semibold")
    ax.set_xlabel("mean paired attributes / object", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, xmax * 1.18 if xmax else 1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.savefig(output_dir / "object_attribute_mean_ranked.png", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path, default=DEFAULT_OURS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nlp, nlp_source = load_nlp()
    print(f"Prompt parser: {nlp_source}")

    frames = []
    for spec in load_benchmarks(args.ours):
        print(f"Processing {spec.name} ({len(spec.prompts)} prompts)")
        frames.append(analyze_dataset(spec.name, spec.prompts, nlp))

    object_df = pd.concat(frames, ignore_index=True)
    summary = write_summary(object_df, args.output_dir)
    plot_object_attribute_boxplot(object_df, args.output_dir)
    plot_mean_bar(summary, args.output_dir)

    print("\n--- Object-Level Attribute Summary ---")
    print(summary.to_string(index=False))
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
