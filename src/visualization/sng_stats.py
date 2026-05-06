#!/usr/bin/env python3
"""Compare prompt complexity across T2I benchmarks using only sng_parser."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sng_parser


ROOT = Path("/fs/nexus-projects/scene_graph_sd")
DEFAULT_OURS = ROOT / "DGE-T2I/data/raw/qwen8b_t2i_prompts_aug_v1.json"
DEFAULT_OUTPUT_DIR = ROOT / "DGE-T2I/reports/visualization/sng_stats"
BENCHMARK_CACHE = ROOT / "DGE-T2I/data/raw/benchmarks"

PALETTE = {
    "Ours": "#2878b5",
    "TIIF-Bench": "#8ecae6",
    "TIIF-Bench (long)": "#219ebc",
    "DPGBench": "#ffb703",
    "T2I-CompBench++": "#fb8500",
    "GenAI-Bench": "#90be6d",
    "GenEval": "#43aa8b",
    "GenEval2": "#4d908e",
    "LongT2IBench": "#9b5de5",
    "GECKONUM": "#adb5bd",
}


@dataclass
class DatasetSpec:
    name: str
    prompts: list[str]
    source: str


def load_jsonl(path: Path, field: str) -> list[str]:
    if not path.exists():
        return []
    prompts = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            value = json.loads(line).get(field)
            if value:
                prompts.append(str(value))
    return prompts


def load_json_prompts(path: Path, field: str = "prompt") -> list[str]:
    if not path.exists():
        return []
    data = json.load(path.open())
    records = data.values() if isinstance(data, dict) else data
    return [str(item[field]) for item in records if isinstance(item, dict) and item.get(field)]


def load_csv_prompts(path: Path) -> list[str]:
    if not path.exists():
        return []
    return pd.read_csv(path)["prompt"].dropna().astype(str).tolist()


def load_t2i_compbench() -> list[str]:
    files = list((ROOT / "T2I-CompBench").glob("**/*val*.parquet"))
    if not files:
        return []
    frames = [pd.read_parquet(path) for path in files]
    if not frames:
        return []
    return pd.concat(frames)["text"].dropna().astype(str).drop_duplicates().tolist()


def load_ours(path: Path) -> DatasetSpec:
    records = json.load(path.open())
    prompts = [str(item["prompt"]) for item in records if item.get("prompt")]
    return DatasetSpec("Ours", prompts, str(path))


def load_longt2ibench(path: Path | None = None) -> list[str]:
    path = path or (BENCHMARK_CACHE / "longt2ibench_LongPrompt-3K.json")
    if not path.exists():
        return []
    data = json.load(path.open())
    prompts = []
    for item in data:
        if not isinstance(item, dict):
            continue
        prompt = item.get("prompt_text") or item.get("en_prompt_text") or item.get("prompt")
        if prompt:
            prompts.append(str(prompt))
    return prompts


def load_geneval() -> list[str]:
    jsonl_path = ROOT / "geneval/prompts/evaluation_metadata.jsonl"
    if jsonl_path.exists():
        return load_jsonl(jsonl_path, "prompt")
    txt_path = ROOT / "geneval/prompts/generation_prompts.txt"
    if txt_path.exists():
        return [line.strip() for line in txt_path.open() if line.strip()]
    return []


def load_geneval2(path: Path | None = None) -> list[str]:
    path = path or (BENCHMARK_CACHE / "geneval2_data.jsonl")
    return load_jsonl(path, "prompt")


def load_benchmarks(ours_path: Path) -> list[DatasetSpec]:
    tiif_path = ROOT / "TIIF-Bench/data/test_prompts/all_prompts.jsonl"
    specs = [
        DatasetSpec("TIIF-Bench", load_jsonl(tiif_path, "short_description"), str(tiif_path)),
        DatasetSpec("TIIF-Bench (long)", load_jsonl(tiif_path, "long_description"), str(tiif_path)),
        DatasetSpec("DPGBench", load_jsonl(ROOT / "ELLA/dpg_bench/prompts.jsonl", "prompt"), str(ROOT / "ELLA/dpg_bench/prompts.jsonl")),
        DatasetSpec("T2I-CompBench++", load_t2i_compbench(), str(ROOT / "T2I-CompBench/**/*val*.parquet")),
        DatasetSpec("GenAI-Bench", load_json_prompts(ROOT / "GenAI-Bench/genai_image.json"), str(ROOT / "GenAI-Bench/genai_image.json")),
        DatasetSpec("GenEval", load_geneval(), str(ROOT / "geneval/prompts/evaluation_metadata.jsonl")),
        DatasetSpec("GenEval2", load_geneval2(), str(BENCHMARK_CACHE / "geneval2_data.jsonl")),
        DatasetSpec("LongT2IBench", load_longt2ibench(), str(BENCHMARK_CACHE / "longt2ibench_LongPrompt-3K.json")),
        DatasetSpec("GECKONUM", load_csv_prompts(ROOT / "GECKONUM/prompts.csv"), str(ROOT / "GECKONUM/prompts.csv")),
        load_ours(ours_path),
    ]
    return [spec for spec in specs if spec.prompts]


def summarize(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def parse_prompt(prompt: str) -> dict:
    try:
        return sng_parser.parse(prompt)
    except Exception as exc:
        print(f"sng_parser failed on prompt: {prompt[:120]!r}; {exc}")
        return {"entities": [], "relations": []}


def analyze_dataset(spec: DatasetSpec) -> tuple[dict, pd.DataFrame, Counter, Counter, Counter]:
    rows = []
    object_terms = Counter()
    attribute_terms = Counter()
    relation_terms = Counter()

    for i, prompt in enumerate(spec.prompts):
        sg = parse_prompt(prompt)
        entities = sg.get("entities", [])
        relations = sg.get("relations", [])
        entity_count = len(entities)
        relation_count = len(relations)
        modifier_count = 0
        entities_with_modifiers = 0

        for entity in entities:
            head = entity.get("lemma_head") or entity.get("head") or entity.get("span")
            if head:
                object_terms[str(head).lower()] += 1
            modifiers = entity.get("modifiers", []) or []
            modifier_count += len(modifiers)
            if modifiers:
                entities_with_modifiers += 1
            for modifier in modifiers:
                if isinstance(modifier, dict):
                    word = modifier.get("lemma_span") or modifier.get("span") or modifier.get("text")
                else:
                    word = str(modifier)
                if word:
                    attribute_terms[str(word).lower()] += 1

        for relation in relations:
            label = relation.get("relation")
            if label:
                relation_terms[str(label).lower()] += 1

        rows.append(
            {
                "dataset": spec.name,
                "prompt_index": i,
                "prompt": prompt,
                "objects_per_prompt": entity_count,
                "attributes_per_prompt": modifier_count,
                "attributes_per_object": modifier_count / entity_count if entity_count else 0.0,
                "entities_with_attributes": entities_with_modifiers,
                "relations_per_prompt": relation_count,
                "relations_per_object": relation_count / entity_count if entity_count else 0.0,
            }
        )

    detail_df = pd.DataFrame(rows)
    total_entities = detail_df["objects_per_prompt"].sum()
    total_modifiers = detail_df["attributes_per_prompt"].sum()
    total_entities_with_modifiers = detail_df["entities_with_attributes"].sum()
    total_relations = detail_df["relations_per_prompt"].sum()

    summary = {
        "Dataset": spec.name,
        "Prompts": len(spec.prompts),
        "Source": "sng_parser",
        "Unique Objects": len(object_terms),
        "Unique Attributes": len(attribute_terms),
        "Unique Relations": len(relation_terms),
        "Avg Attributes/Entity": (total_modifiers / total_entities) if total_entities else 0.0,
        "Entities w/Attributes": int(total_entities_with_modifiers),
        "Pct Entities w/Attributes": (total_entities_with_modifiers / total_entities * 100) if total_entities else 0.0,
        "Avg Relations/Object": (total_relations / total_entities) if total_entities else 0.0,
    }
    for prefix, column in [
        ("Objects/Prompt", detail_df["objects_per_prompt"]),
        ("Attributes/Prompt", detail_df["attributes_per_prompt"]),
        ("Attributes/Object", detail_df["attributes_per_object"]),
        ("Entities w/Attributes", detail_df["entities_with_attributes"]),
        ("Relations/Prompt", detail_df["relations_per_prompt"]),
        ("Relations/Object", detail_df["relations_per_object"]),
    ]:
        for stat, value in summarize(column).items():
            summary[f"{prefix} {stat}"] = value

    return summary, detail_df, object_terms, attribute_terms, relation_terms


def ordered_groups(detail_df: pd.DataFrame, column: str) -> tuple[list[str], list[np.ndarray]]:
    stats = detail_df.groupby("dataset")[column].agg(["mean", "median"]).sort_values(
        ["mean", "median"], ascending=False
    )
    labels = stats.index.tolist()
    data = [detail_df.loc[detail_df["dataset"] == label, column].to_numpy() for label in labels]
    return labels, data


def style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=19, fontweight="semibold", pad=12)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis="x", labelrotation=30, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def draw_boxplot(ax, labels: list[str], data: list[np.ndarray], title: str, ylabel: str) -> None:
    colors = [PALETTE.get(label, "#cccccc") for label in labels]
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
    for patch, color, label in zip(box["boxes"], colors, labels):
        patch.set_facecolor(color)
        patch.set_alpha(0.78 if label != "Ours" else 0.95)
        if label == "Ours":
            patch.set_linewidth(2.4)

    means = [values.mean() if len(values) else 0.0 for values in data]
    ax.scatter(
        np.arange(1, len(labels) + 1),
        means,
        marker="D",
        s=58,
        color="#1b9e77",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label="Mean",
    )
    if "Ours" in labels:
        ours_idx = labels.index("Ours") + 1
        ax.annotate(
            f"Ours\nmean {means[ours_idx - 1]:.2f}",
            xy=(ours_idx, means[ours_idx - 1]),
            xytext=(ours_idx + 0.35, means[ours_idx - 1] + max(means) * 0.08 + 0.05),
            arrowprops={"arrowstyle": "->", "linewidth": 1.2, "color": "#333333"},
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#555555", "alpha": 0.95},
            fontsize=11,
        )
    style_axes(ax, title, ylabel)


def plot_boxplots(detail_df: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("objects_per_prompt", "Objects per Prompt", "Objects"),
        ("attributes_per_prompt", "Attributes per Prompt", "Attributes"),
        ("attributes_per_object", "Attributes per Object", "Attributes / Object"),
        ("relations_per_prompt", "Relations per Prompt", "Relations"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    for ax, (column, title, ylabel) in zip(axes.ravel(), metrics):
        labels, data = ordered_groups(detail_df, column)
        draw_boxplot(ax, labels, data, title, ylabel)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=1, frameon=False, fontsize=13)
    fig.suptitle("sng_parser Prompt Complexity Distributions", fontsize=24, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_dir / "sng_distribution_boxplots.png", dpi=240)
    plt.close(fig)


def plot_ranked_attributes_per_object(detail_df: pd.DataFrame, output_dir: Path) -> None:
    labels, data = ordered_groups(detail_df, "attributes_per_object")
    fig, ax = plt.subplots(figsize=(15, 8))
    draw_boxplot(ax, labels, data, "Attributes per Object", "Attributes / Object")
    means = [values.mean() if len(values) else 0.0 for values in data]
    if "Ours" in labels:
        ours_idx = labels.index("Ours")
        top_label = labels[0]
        top_mean = means[0]
        ours_mean = means[ours_idx]
        ax.text(
            0.02,
            0.96,
            f"Ordered by mean. Top: {top_label} ({top_mean:.2f}); Ours: {ours_mean:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=14,
            bbox={"boxstyle": "round,pad=0.35", "fc": "#fff7e6", "ec": "#b56b00", "alpha": 0.95},
        )
    fig.tight_layout()
    fig.savefig(output_dir / "sng_attributes_per_object_boxplot_ranked.png", dpi=240)
    plt.close(fig)


def plot_histograms(detail_df: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("objects_per_prompt", "Objects per Prompt"),
        ("attributes_per_prompt", "Attributes per Prompt"),
        ("attributes_per_object", "Attributes per Object"),
        ("relations_per_prompt", "Relations per Prompt"),
    ]
    for column, title in metrics:
        fig, ax = plt.subplots(figsize=(11, 6))
        for dataset, group in detail_df.groupby("dataset"):
            ax.hist(group[column], bins=28, alpha=0.35, density=True, label=dataset, color=PALETTE.get(dataset))
        style_axes(ax, f"sng_parser: {title}", title)
        ax.set_ylabel("Density", fontsize=15)
        ax.legend(fontsize=9, frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / f"sng_{column}.png", dpi=220)
        plt.close(fig)


def write_top_terms(
    output_dir: Path,
    object_terms: dict[str, Counter],
    attribute_terms: dict[str, Counter],
    relation_terms: dict[str, Counter],
) -> None:
    rows = []
    for term_type, counters in [
        ("object", object_terms),
        ("attribute", attribute_terms),
        ("relation", relation_terms),
    ]:
        for dataset, counter in counters.items():
            for term, count in counter.most_common(50):
                if term:
                    rows.append({"dataset": dataset, "type": term_type, "term": term, "count": count})
    pd.DataFrame(rows).to_csv(output_dir / "sng_top_terms.csv", index=False)


def write_analysis(summary_df: pd.DataFrame, output_dir: Path) -> None:
    ours = summary_df.loc[summary_df["Dataset"] == "Ours"].iloc[0]
    others = summary_df[summary_df["Dataset"] != "Ours"]
    lines = [
        "# sng_parser Prompt Complexity Analysis",
        "",
        "All datasets are parsed from prompt text with `sng_parser`. Attributes are parser modifiers, entities are parser entities, and relations are parser relations.",
        "",
        "## Ours",
        "",
        f"- Prompts: {int(ours['Prompts'])}",
        f"- Avg attributes/entity: {ours['Avg Attributes/Entity']:.2f}",
        f"- Entities with attributes: {ours['Pct Entities w/Attributes']:.2f}%",
        f"- Avg relations/object: {ours['Avg Relations/Object']:.2f}",
        f"- Objects/prompt: {ours['Objects/Prompt mean']:.2f}",
        f"- Relations/prompt: {ours['Relations/Prompt mean']:.2f}",
        "",
        "## Comparison Notes",
        "",
        f"- Ours has higher attributes/entity than {(others['Avg Attributes/Entity'] < ours['Avg Attributes/Entity']).sum()} of {len(others)} comparison datasets.",
        f"- Ours has higher percent entities with attributes than {(others['Pct Entities w/Attributes'] < ours['Pct Entities w/Attributes']).sum()} of {len(others)} comparison datasets.",
        f"- Ours has higher relations/object than {(others['Avg Relations/Object'] < ours['Avg Relations/Object']).sum()} of {len(others)} comparison datasets.",
        "",
        "See `sng_summary.csv`, `sng_per_prompt_distributions.csv`, and the PNG plots in this directory.",
    ]
    (output_dir / "sng_analysis.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path, default=DEFAULT_OURS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    details = []
    object_terms = {}
    attribute_terms = {}
    relation_terms = {}

    for spec in load_benchmarks(args.ours):
        print(f"Processing {spec.name} ({len(spec.prompts)} prompts)")
        summary, detail_df, objects, attributes, relations = analyze_dataset(spec)
        summaries.append(summary)
        details.append(detail_df)
        object_terms[spec.name] = objects
        attribute_terms[spec.name] = attributes
        relation_terms[spec.name] = relations

    summary_df = pd.DataFrame(summaries)
    detail_df = pd.concat(details, ignore_index=True)

    summary_cols = [
        "Dataset",
        "Avg Attributes/Entity",
        "Pct Entities w/Attributes",
        "Avg Relations/Object",
    ]
    summary_df[summary_cols].to_csv(args.output_dir / "sng_summary.csv", index=False)
    detail_df.to_csv(args.output_dir / "sng_per_prompt_distributions.csv", index=False)
    write_top_terms(args.output_dir, object_terms, attribute_terms, relation_terms)
    plot_boxplots(detail_df, args.output_dir)
    plot_ranked_attributes_per_object(detail_df, args.output_dir)
    plot_histograms(detail_df, args.output_dir)
    write_analysis(summary_df, args.output_dir)

    display_cols = [
        "Dataset",
        "Prompts",
        "Avg Attributes/Entity",
        "Pct Entities w/Attributes",
        "Avg Relations/Object",
        "Objects/Prompt mean",
        "Attributes/Prompt mean",
        "Relations/Prompt mean",
    ]
    print("\n--- sng_parser Summary ---")
    print(summary_df[display_cols].sort_values("Avg Attributes/Entity", ascending=False).to_string(index=False))
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
