#!/usr/bin/env python3
"""Plot prompt nuance metrics across T2I benchmark datasets using spaCy."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.pos_stats import (
    BENCHMARK_COLORS,
    DEFAULT_OURS,
    ROOT,
    load_benchmarks,
    load_nlp,
)

DEFAULT_OUTPUT_DIR = ROOT / "DGE-T2I/reports/visualization/nuance_stats"


def dependency_depth(token) -> int:
    children = list(token.children)
    if not children:
        return 1
    return 1 + max(dependency_depth(child) for child in children)


def doc_metrics(doc) -> dict[str, float | set[str]]:
    tokens = [token for token in doc if not token.is_space and not token.is_punct]
    token_count = len(tokens)
    objects = [token for token in tokens if token.pos_ in {"NOUN", "PROPN"}]
    attrs = [token for token in tokens if token.pos_ == "ADJ"]
    rels = [token for token in tokens if token.pos_ in {"VERB", "ADP"}]
    noun_chunks = list(doc.noun_chunks)
    prep_phrases = [token for token in tokens if token.dep_ == "prep" or token.pos_ == "ADP"]
    conjunctions = [token for token in tokens if token.pos_ in {"CCONJ", "SCONJ"}]
    roots = [token for token in doc if token.head == token]
    max_depth = max((dependency_depth(root) for root in roots), default=0)

    object_count = len(objects)
    attr_count = len(attrs)
    rel_count = len(rels)
    chunk_count = len(noun_chunks)
    atom_count = object_count + attr_count + rel_count

    return {
        "tokens": token_count,
        "objects": object_count,
        "attributes": attr_count,
        "relations": rel_count,
        "relations_per_object": rel_count / object_count if object_count else 0.0,
        "attributes_per_object": attr_count / object_count if object_count else 0.0,
        "noun_chunks": chunk_count,
        "noun_chunks_per_prompt": chunk_count,
        "adjectives_per_noun_chunk": attr_count / chunk_count if chunk_count else 0.0,
        "prep_phrases": len(prep_phrases),
        "conjunctions": len(conjunctions),
        "dependency_depth": max_depth,
        "atoms_per_20_tokens": atom_count / token_count * 20 if token_count else 0.0,
        "objects_per_20_tokens": object_count / token_count * 20 if token_count else 0.0,
        "attributes_per_20_tokens": attr_count / token_count * 20 if token_count else 0.0,
        "relations_per_20_tokens": rel_count / token_count * 20 if token_count else 0.0,
        "unique_attributes": {token.lemma_.lower() for token in attrs if not token.is_stop},
        "unique_relations": {token.lemma_.lower() for token in rels if not token.is_stop},
    }


def analyze_dataset(name: str, prompts: list[str], nlp) -> tuple[dict, pd.DataFrame]:
    rows = []
    attr_vocab = Counter()
    rel_vocab = Counter()

    for i, doc in enumerate(nlp.pipe(prompts, batch_size=256)):
        metrics = doc_metrics(doc)
        attr_vocab.update(metrics.pop("unique_attributes"))
        rel_vocab.update(metrics.pop("unique_relations"))
        metrics.update({"dataset": name, "prompt_index": i, "prompt": prompts[i]})
        rows.append(metrics)

    df = pd.DataFrame(rows)
    summary = {
        "Dataset": name,
        "Prompts": len(prompts),
        "Unique Attributes / 100 Prompts": len(attr_vocab) / len(prompts) * 100 if prompts else 0.0,
        "Unique Relations / 100 Prompts": len(rel_vocab) / len(prompts) * 100 if prompts else 0.0,
    }
    mean_cols = [
        "tokens",
        "relations_per_object",
        "attributes_per_object",
        "noun_chunks_per_prompt",
        "adjectives_per_noun_chunk",
        "prep_phrases",
        "conjunctions",
        "dependency_depth",
        "atoms_per_20_tokens",
        "objects_per_20_tokens",
        "attributes_per_20_tokens",
        "relations_per_20_tokens",
    ]
    for col in mean_cols:
        summary[col] = df[col].mean() if not df.empty else 0.0
    return summary, df


def add_nuance_score(summary_df: pd.DataFrame) -> pd.DataFrame:
    score_cols = [
        "relations_per_object",
        "attributes_per_object",
        "noun_chunks_per_prompt",
        "adjectives_per_noun_chunk",
        "prep_phrases",
        "dependency_depth",
        "atoms_per_20_tokens",
        "Unique Attributes / 100 Prompts",
        "Unique Relations / 100 Prompts",
    ]
    z = summary_df[score_cols].copy()
    z = (z - z.mean()) / z.std(ddof=0).replace(0, np.nan)
    summary_df["Nuance Score"] = z.mean(axis=1)
    return summary_df


def style_ranked_bar(ax, title: str, xlabel: str) -> None:
    ax.set_title(title, fontsize=17, fontweight="semibold", pad=10)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def ranked_bar(summary_df: pd.DataFrame, column: str, title: str, xlabel: str, output_path: Path) -> None:
    ranked = summary_df.sort_values(column, ascending=True)
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    ax.barh(
        ranked["Dataset"],
        ranked[column],
        color=[BENCHMARK_COLORS.get(name, "#cccccc") for name in ranked["Dataset"]],
        edgecolor=["#111111" if name == "Ours" else "white" for name in ranked["Dataset"]],
        linewidth=[2.2 if name == "Ours" else 0.8 for name in ranked["Dataset"]],
        alpha=0.92,
    )
    xmax = ranked[column].max()
    for y, value in enumerate(ranked[column]):
        ax.text(value + xmax * 0.015, y, f"{value:.2f}", va="center", fontsize=10)
    ax.set_xlim(0, xmax * 1.18 if xmax > 0 else 1)
    style_ranked_bar(ax, title, xlabel)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def boxplot(per_prompt_df: pd.DataFrame, column: str, title: str, ylabel: str, output_path: Path) -> None:
    stats = per_prompt_df.groupby("dataset")[column].agg(["mean", "median"]).sort_values(
        ["mean", "median"], ascending=False
    )
    labels = stats.index.tolist()
    data = [per_prompt_df.loc[per_prompt_df["dataset"] == label, column].to_numpy() for label in labels]
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
    ax.scatter(np.arange(1, len(labels) + 1), means, marker="D", s=54, color="#1b9e77", edgecolor="white", linewidth=0.8, zorder=3, label="Mean")
    ax.set_title(title, fontsize=18, fontweight="semibold", pad=10)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel("Dataset (ordered by mean)", fontsize=14)
    ax.tick_params(axis="x", rotation=30, labelsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_all(summary_df: pd.DataFrame, per_prompt_df: pd.DataFrame, output_dir: Path) -> None:
    ranked_bar(summary_df, "Unique Attributes / 100 Prompts", "Unique Attribute Types per 100 Prompts", "unique adjective lemmas / 100 prompts", output_dir / "unique_attributes_per_100_prompts.png")
    ranked_bar(summary_df, "Unique Relations / 100 Prompts", "Unique Relation Types per 100 Prompts", "unique relation lemmas / 100 prompts", output_dir / "unique_relations_per_100_prompts.png")
    ranked_bar(summary_df, "atoms_per_20_tokens", "Compositional Density per 20 Tokens", "Compositional Atoms per 20 Tokens", output_dir / "density_per_20_tokens_ranked.png")
    ranked_bar(summary_df, "Nuance Score", "Composite Nuance Score", "mean z-score across nuance metrics", output_dir / "nuance_score_ranked.png")
    boxplot(per_prompt_df, "noun_chunks_per_prompt", "Noun Chunks per Prompt", "noun chunks", output_dir / "noun_chunks_per_prompt_boxplot.png")
    boxplot(per_prompt_df, "adjectives_per_noun_chunk", "Adjectives per Noun Chunk", "adjectives / noun chunk", output_dir / "adjectives_per_noun_chunk_boxplot.png")
    boxplot(per_prompt_df, "prep_phrases", "Prepositional Phrases per Prompt", "prep phrases", output_dir / "prep_phrases_per_prompt_boxplot.png")
    boxplot(per_prompt_df, "conjunctions", "Conjunctions per Prompt", "conjunctions", output_dir / "conjunctions_per_prompt_boxplot.png")
    boxplot(per_prompt_df, "dependency_depth", "Dependency Parse Depth", "max dependency depth", output_dir / "dependency_depth_boxplot.png")



def tradeoff_scatter(
    summary_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    sizes = np.sqrt(summary_df["Prompts"]) * 18
    for _, row in summary_df.iterrows():
        dataset = row["Dataset"]
        ax.scatter(
            row[x_col],
            row[y_col],
            s=sizes.loc[row.name],
            color=BENCHMARK_COLORS.get(dataset, "#cccccc"),
            edgecolor="#111111" if dataset == "Ours" else "white",
            linewidth=2.2 if dataset == "Ours" else 0.8,
            alpha=0.9,
            zorder=3 if dataset == "Ours" else 2,
        )
        offset_x = summary_df[x_col].max() * (0.012 if dataset != "Ours" else 0.018)
        offset_y = summary_df[y_col].max() * (0.012 if dataset != "Ours" else 0.018)
        ax.text(
            row[x_col] + offset_x,
            row[y_col] + offset_y,
            dataset,
            fontsize=12 if dataset == "Ours" else 9,
            fontweight="bold" if dataset == "Ours" else "normal",
            color="#111111",
        )
    ax.set_title(title, fontsize=19, fontweight="semibold", pad=12)
    ax.set_xlabel("Mean Prompt Length", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_tradeoff_scatters(summary_df: pd.DataFrame, output_dir: Path) -> None:
    tradeoff_scatter(
        summary_df,
        "tokens",
        "relations_per_object",
        "Prompt Length vs. Relation Density",
        "Relations per Object",
        output_dir / "prompt_length_vs_relations_per_object.png",
    )
    tradeoff_scatter(
        summary_df,
        "tokens",
        "atoms_per_20_tokens",
        "Prompt Length vs. Compositional Information Density",
        "Compositional Atoms per 20 Tokens",
        output_dir / "prompt_length_vs_compositional_density.png",
    )
    tradeoff_scatter(
        summary_df,
        "tokens",
        "attributes_per_object",
        "Prompt Length vs. Attribute Density",
        "Attributes per Object",
        output_dir / "prompt_length_vs_attributes_per_object.png",
    )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path, default=DEFAULT_OURS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nlp, nlp_source = load_nlp()
    print(f"Prompt parser: {nlp_source}")

    summaries = []
    detail_frames = []
    for spec in load_benchmarks(args.ours):
        print(f"Processing {spec.name} ({len(spec.prompts)} prompts)")
        summary, detail_df = analyze_dataset(spec.name, spec.prompts, nlp)
        summaries.append(summary)
        detail_frames.append(detail_df)

    summary_df = add_nuance_score(pd.DataFrame(summaries))
    per_prompt_df = pd.concat(detail_frames, ignore_index=True)

    summary_df.to_csv(args.output_dir / "nuance_summary.csv", index=False)
    per_prompt_df.to_csv(args.output_dir / "nuance_per_prompt.csv", index=False)
    plot_all(summary_df, per_prompt_df, args.output_dir)
    plot_tradeoff_scatters(summary_df, args.output_dir)

    display_cols = [
        "Dataset",
        "Nuance Score",
        "atoms_per_20_tokens",
        "relations_per_object",
        "Unique Attributes / 100 Prompts",
        "Unique Relations / 100 Prompts",
        "noun_chunks_per_prompt",
        "prep_phrases",
    ]
    print("\n--- Nuance Summary ---")
    print(summary_df[display_cols].sort_values("Nuance Score", ascending=False).to_string(index=False))
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
