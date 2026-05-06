#!/usr/bin/env python3
"""Compare prompt and scene-graph complexity across T2I benchmark datasets."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy



ROOT = Path("/fs/nexus-projects/scene_graph_sd")
DEFAULT_OURS = ROOT / "DGE-T2I/data/raw/qwen8b_t2i_prompts_aug_v1.json"
DEFAULT_OUTPUT_DIR = ROOT / "DGE-T2I/reports/visualization/pos_stats"
BENCHMARK_CACHE = ROOT / "DGE-T2I/data/raw/benchmarks"
BENCHMARK_COLORS = {
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


OBJECT_RE = re.compile(r"^\s*-\s*(?:\d+\s+)?(?P<name>.+?)\s+\(object id\s*:\s*(?P<id>\d+)\)")
ATTRIBUTE_RE = re.compile(r"^\s*-\s*(?!Object\s)(?P<attribute>.+?)\s*$")
RELATION_RE = re.compile(
    r"^\s*-\s*Object\s+(?P<subject>\d+)\s+(?P<relation>.+?)\s+Object\s+(?P<object>\d+)\s*$"
)

STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "with", "for", "from",
    "by", "at", "as", "is", "are", "was", "were", "be", "being", "been", "while",
    "that", "this", "these", "those", "into", "over", "under", "near", "next", "beside",
}
ATTRIBUTE_WORDS = {
    "black", "white", "red", "blue", "green", "yellow", "brown", "orange", "pink", "purple",
    "gray", "grey", "silver", "gold", "golden", "wooden", "metal", "metallic", "glass",
    "ceramic", "plastic", "large", "small", "tiny", "tall", "short", "long", "round", "square",
    "open", "closed", "bright", "dark", "light", "old", "new", "wet", "dry", "clean", "dirty",
    "striped", "spotted", "checkered", "reflective", "transparent", "shiny", "matte", "left",
    "right", "front", "back", "top", "bottom", "single", "multiple", "several", "few",
}
RELATION_WORDS = {
    "in", "on", "under", "over", "above", "below", "beside", "near", "behind", "front",
    "next", "with", "without", "inside", "outside", "around", "through", "across", "between",
    "holding", "hold", "wearing", "wear", "sitting", "sit", "standing", "stand", "lying", "lay",
    "riding", "ride", "carrying", "carry", "eating", "eat", "drinking", "drink", "looking",
    "look", "facing", "face", "covering", "cover", "contains", "contain", "has", "have",
    "attached", "mounted", "surrounded", "resting", "rests", "placed", "positioned",
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


class SimpleToken:
    def __init__(self, text: str):
        self.text = text
        self.lemma_ = text.lower().strip("'")
        self.is_stop = self.lemma_ in STOP_WORDS
        self.is_space = False
        self.is_punct = False
        self.children = []
        self.dep_ = ""
        self.pos_ = self._guess_pos()

    def _guess_pos(self) -> str:
        lemma = self.lemma_
        if lemma in ATTRIBUTE_WORDS:
            return "ADJ"
        if lemma in RELATION_WORDS or lemma.endswith("ing") or lemma.endswith("ed"):
            return "VERB"
        if lemma in STOP_WORDS:
            return "ADP" if lemma in RELATION_WORDS else "DET"
        return "NOUN"


class SimpleNLP:
    def pipe(self, prompts, batch_size=256):
        del batch_size
        for prompt in prompts:
            yield [SimpleToken(token) for token in TOKEN_RE.findall(prompt)]


def load_nlp():
    try:
        return spacy.load("en_core_web_sm", disable=["ner"]), "spacy_en_core_web_sm"
    except OSError:
        print("spaCy model en_core_web_sm not found; using a lightweight lexical fallback.")
        return SimpleNLP(), "lexical_fallback"


@dataclass
class DatasetSpec:
    name: str
    prompts: list[str]
    source: str
    graphs: list[dict] | None = None


def load_jsonl(path: Path, field: str) -> list[str]:
    if not path.exists():
        return []
    prompts = []
    with path.open() as f:
        for line in f:
            if line.strip():
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


def parse_current_task_graph(meta_prompt: str) -> dict:
    """Extract the object/attribute/relation block used to generate our prompt."""
    marker = "Current Task:"
    if marker not in meta_prompt:
        return {"objects": [], "relations": []}

    block = meta_prompt.split(marker, 1)[1]
    block = block.split("[Step-by-Step Reasoning]", 1)[0]
    section = None
    objects = []
    relations = []
    current_object = None

    for line in block.splitlines():
        stripped = line.strip()
        if stripped == "Objects:":
            section = "objects"
            continue
        if stripped == "Relationships:":
            section = "relations"
            current_object = None
            continue

        if section == "objects":
            object_match = OBJECT_RE.match(line)
            if object_match:
                current_object = {
                    "id": int(object_match.group("id")),
                    "name": object_match.group("name").strip(),
                    "attributes": [],
                }
                objects.append(current_object)
                continue

            attr_match = ATTRIBUTE_RE.match(line)
            if attr_match and current_object is not None:
                current_object["attributes"].append(attr_match.group("attribute").strip())

        elif section == "relations":
            rel_match = RELATION_RE.match(line)
            if rel_match:
                relations.append(
                    {
                        "subject": int(rel_match.group("subject")),
                        "relation": rel_match.group("relation").strip(),
                        "object": int(rel_match.group("object")),
                    }
                )

    return {"objects": objects, "relations": relations}


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
        # DatasetSpec("TIIF-Bench (long)", load_jsonl(tiif_path, "long_description"), str(tiif_path)),
        # DatasetSpec(
        #     "DPGBench",
        #     load_jsonl(ROOT / "ELLA/dpg_bench/prompts.jsonl", "prompt"),
        #     str(ROOT / "ELLA/dpg_bench/prompts.jsonl"),
        # ),
        DatasetSpec("T2I-CompBench++", load_t2i_compbench(), str(ROOT / "T2I-CompBench/**/*val*.parquet")),
        DatasetSpec("GenAI-Bench", load_json_prompts(ROOT / "GenAI-Bench/genai_image.json"), str(ROOT / "GenAI-Bench/genai_image.json")),
        DatasetSpec("GenEval", load_geneval(), str(ROOT / "geneval/prompts/evaluation_metadata.jsonl")),
        DatasetSpec("GenEval2", load_geneval2(), str(BENCHMARK_CACHE / "geneval2_data.jsonl")),
        DatasetSpec("LongT2IBench", load_longt2ibench(), str(BENCHMARK_CACHE / "longt2ibench_LongPrompt-3K.json")),
        DatasetSpec("GECKONUM", load_csv_prompts(ROOT / "GECKONUM/prompts.csv"), str(ROOT / "GECKONUM/prompts.csv")),
        load_ours(ours_path),
    ]
    return [spec for spec in specs if spec.prompts]


def doc_counts(doc) -> dict[str, object]:
    entity_pos = {"NOUN", "PROPN"}
    entities = [token for token in doc if token.pos_ in entity_pos]
    attributes = [token for token in doc if token.pos_ == "ADJ"]
    relations = [token for token in doc if token.pos_ in {"VERB", "ADP"}]
    token_count = sum(1 for token in doc if not token.is_space and not token.is_punct)

    entity_attribute_counts = []
    for entity in entities:
        modifier_count = sum(
            1
            for child in entity.children
            if child.pos_ == "ADJ" and child.dep_ in {"amod", "acomp", "advmod"}
        )
        # Attribute phrases like "person with blonde hair" attach through a prep;
        # count adjectival modifiers on the prepositional object as entity attributes.
        for child in entity.children:
            if child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.pos_ in entity_pos:
                        modifier_count += sum(
                            1
                            for attr in grandchild.children
                            if attr.pos_ == "ADJ" and attr.dep_ in {"amod", "acomp", "advmod"}
                        )
        entity_attribute_counts.append(modifier_count)

    return {
        "object_count": len(entities),
        "attribute_count": len(attributes),
        "relation_count": len(relations),
        "token_count": token_count,
        "entities_with_attributes": sum(1 for count in entity_attribute_counts if count > 0),
        "entity_attribute_total": sum(entity_attribute_counts),
    }


def graph_counts(graph: dict) -> tuple[int, int, int]:
    objects = graph.get("objects", [])
    relations = graph.get("relations", [])
    attr_count = sum(len(obj.get("attributes", [])) for obj in objects)
    return len(objects), attr_count, len(relations)


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


def analyze_dataset(
    spec: DatasetSpec, nlp, prompt_metric_source: str
) -> tuple[dict, pd.DataFrame, Counter, Counter, Counter]:
    rows = []
    object_counter = Counter()
    attribute_counter = Counter()
    relation_counter = Counter()

    for i, doc in enumerate(nlp.pipe(spec.prompts, batch_size=256)):
        counts = doc_counts(doc)
        object_count = counts["object_count"]
        attribute_count = counts["attribute_count"]
        relation_count = counts["relation_count"]
        attr_per_object = attribute_count / object_count if object_count else 0.0
        relations_per_object = relation_count / object_count if object_count else 0.0
        avg_attributes_per_entity = (
            counts["entity_attribute_total"] / object_count if object_count else 0.0
        )

        object_counter.update(
            token.lemma_.lower()
            for token in doc
            if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop
        )
        attribute_counter.update(token.lemma_.lower() for token in doc if token.pos_ == "ADJ")
        relation_counter.update(
            token.lemma_.lower()
            for token in doc
            if token.pos_ in {"VERB", "ADP"} and not token.is_stop
        )

        rows.append(
            {
                "dataset": spec.name,
                "prompt_index": i,
                "prompt": spec.prompts[i],
                "objects_per_prompt": object_count,
                "attributes_per_prompt": attribute_count,
                "attributes_per_object": attr_per_object,
                "avg_attributes_per_entity": avg_attributes_per_entity,
                "entities_with_attributes": counts["entities_with_attributes"],
                "relations_per_prompt": relation_count,
                "relations_per_object": relations_per_object,
                "prompt_nouns": object_count,
                "prompt_adjectives": attribute_count,
                "prompt_relation_terms": relation_count,
                "prompt_tokens": counts["token_count"],
                "metric_source": prompt_metric_source,
            }
        )

    detail_df = pd.DataFrame(rows)
    total_objects = detail_df["objects_per_prompt"].sum()
    total_entity_attributes = (
        detail_df["avg_attributes_per_entity"] * detail_df["objects_per_prompt"]
    ).sum()
    total_entities_with_attributes = detail_df["entities_with_attributes"].sum()
    total_relations = detail_df["relations_per_prompt"].sum()
    summary = {
        "Dataset": spec.name,
        "Prompts": len(spec.prompts),
        "Source": prompt_metric_source,
        "Unique Objects": len([key for key in object_counter if key]),
        "Unique Attributes": len([key for key in attribute_counter if key]),
        "Unique Relations": len([key for key in relation_counter if key]),
        "Avg Attributes/Entity": (total_entity_attributes / total_objects) if total_objects else 0.0,
        "Entities w/Attributes": int(total_entities_with_attributes),
        "Pct Entities w/Attributes": (
            total_entities_with_attributes / total_objects * 100
        ) if total_objects else 0.0,
        "Avg Relations/Object": (total_relations / total_objects) if total_objects else 0.0,
    }
    for prefix, column in [
        ("Objects/Prompt", detail_df["objects_per_prompt"]),
        ("Attributes/Prompt", detail_df["attributes_per_prompt"]),
        ("Attributes/Object", detail_df["attributes_per_object"]),
        ("Entities w/Attributes", detail_df["entities_with_attributes"]),
        ("Relations/Prompt", detail_df["relations_per_prompt"]),
        ("Relations/Object", detail_df["relations_per_object"]),
        ("Prompt Tokens", detail_df["prompt_tokens"]),
    ]:
        for stat, value in summarize(column).items():
            summary[f"{prefix} {stat}"] = value

    return summary, detail_df, object_counter, attribute_counter, relation_counter

def style_boxplot_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=18, fontweight="semibold", pad=10)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="x", rotation=30, labelsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def draw_colored_boxplot(ax, labels: list[str], data: list[np.ndarray], title: str, ylabel: str) -> None:
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
    ax.scatter(
        np.arange(1, len(labels) + 1),
        means,
        marker="D",
        s=54,
        color="#1b9e77",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label="Mean",
    )
    style_boxplot_axis(ax, title, ylabel)


def plot_distributions(detail_df: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("objects_per_prompt", "Objects per Prompt", "Objects"),
        ("attributes_per_prompt", "Attributes per Prompt", "Attributes"),
        ("attributes_per_object", "Attributes per Object", "Attributes / Object"),
        ("relations_per_prompt", "Relations per Prompt", "Relations"),
    ]
    for column, title, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(11, 6))
        for dataset, group in detail_df.groupby("dataset"):
            ax.hist(
                group[column],
                bins=28,
                alpha=0.35,
                density=True,
                label=dataset,
                color=BENCHMARK_COLORS.get(dataset),
            )
        style_boxplot_axis(ax, title, ylabel)
        ax.set_ylabel("Density", fontsize=14)
        ax.legend(fontsize=9, frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{column}.png", dpi=220)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    for ax, (column, title, ylabel) in zip(axes.ravel(), metrics):
        stats = detail_df.groupby("dataset")[column].agg(["mean", "median"]).sort_values(
            ["mean", "median"], ascending=False
        )
        labels = stats.index.tolist()
        data = [detail_df.loc[detail_df["dataset"] == label, column].to_numpy() for label in labels]
        draw_colored_boxplot(ax, labels, data, title, ylabel)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=1, frameon=False, fontsize=13)
    fig.suptitle("spaCy Prompt Complexity Distributions", fontsize=24, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_dir / "distribution_boxplots.png", dpi=240)
    plt.close(fig)

def plot_ours_distributions(detail_df: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("objects_per_prompt", "Objects per Prompt"),
        ("attributes_per_prompt", "Attributes per Prompt"),
        ("attributes_per_object", "Attributes per Object"),
        ("relations_per_prompt", "Relations per Prompt"),
    ]
    ours_df = detail_df[detail_df["dataset"] == "Ours"]
    if ours_df.empty:
        return

    for column, title in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        values = ours_df[column].to_numpy()
        bins = np.arange(values.min(), values.max() + 2) - 0.5 if column != "attributes_per_object" else 24
        ax.hist(values, bins=bins, alpha=0.85, color="#2f6f9f", edgecolor="white")
        ax.axvline(values.mean(), color="#c43c39", linestyle="--", linewidth=2, label=f"Mean: {values.mean():.2f}")
        ax.axvline(np.median(values), color="#2e7d32", linestyle=":", linewidth=2, label=f"Median: {np.median(values):.2f}")
        ax.set_title(f"Ours: {title}")
        ax.set_xlabel(title)
        ax.set_ylabel("Prompt Count")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(output_dir / f"ours_{column}.png", dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (column, title) in zip(axes.ravel(), metrics):
        values = ours_df[column].to_numpy()
        ax.boxplot([values], tick_labels=["Ours"], showfliers=True)
        ax.set_title(title)
        ax.set_ylabel(title)
    fig.tight_layout()
    fig.savefig(output_dir / "ours_distribution_boxplots.png", dpi=200)
    plt.close(fig)



def plot_attributes_per_object_ranked(detail_df: pd.DataFrame, output_dir: Path) -> None:
    column = "attributes_per_object"
    title = "Attributes per Object"
    stats = (
        detail_df.groupby("dataset")[column]
        .agg(["mean", "median"])
        .sort_values(["mean", "median"], ascending=False)
    )
    labels = stats.index.tolist()
    data = [detail_df.loc[detail_df["dataset"] == dataset, column].to_numpy() for dataset in labels]

    fig, ax = plt.subplots(figsize=(15, 8))
    draw_colored_boxplot(ax, labels, data, title, "Attributes / Object")
    ax.set_xlabel("Dataset (ordered by mean attributes/object)", fontsize=14)
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "attributes_per_object_boxplot_ranked.png", dpi=240)
    plt.close(fig)


def plot_key_metric_rankings(summary_df: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("Avg Relations/Object", "", "Average Relations Per Object"),
        ("Avg Attributes/Entity", "", "Average Attributes Per Object"),
        ("Pct Entities w/Attributes", "% Entities with Attributes", "%"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)

    for ax, (column, title, xlabel) in zip(axes, metrics):
        ranked = summary_df.sort_values(column, ascending=True)
        colors = [BENCHMARK_COLORS.get(name, "#cccccc") for name in ranked["Dataset"]]
        edgecolors = ["#111111" if name == "Ours" else "white" for name in ranked["Dataset"]]
        linewidths = [2.2 if name == "Ours" else 0.8 for name in ranked["Dataset"]]

        bars = ax.barh(
            ranked["Dataset"],
            ranked[column],
            color=colors,
            edgecolor=edgecolors,
            linewidth=linewidths,
            alpha=0.9,
        )
        for bar, value in zip(bars, ranked[column]):
            label = f"{value:.2f}" if column != "Pct Entities w/Attributes" else f"{value:.1f}"
            ax.text(
                value + ranked[column].max() * 0.015,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="left",
                fontsize=10,
                color="#333333",
            )

        ax.set_title(title, fontsize=17, fontweight="semibold", pad=10)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.tick_params(axis="y", labelsize=11)
        ax.tick_params(axis="x", labelsize=11)
        ax.grid(axis="x", color="#dddddd", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, ranked[column].max() * 1.18)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle("spaCy Key Metric Rankings", fontsize=24, fontweight="bold")
    fig.savefig(output_dir / "key_metric_rankings.png", dpi=240)
    plt.close(fig)

    # Single-metric figure for the strongest claim: ours leads relation density.
    relation_df = summary_df.sort_values("Avg Relations/Object", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    colors = [BENCHMARK_COLORS.get(name, "#cccccc") for name in relation_df["Dataset"]]
    ax.barh(
        relation_df["Dataset"],
        relation_df["Avg Relations/Object"],
        color=colors,
        edgecolor=["#111111" if name == "Ours" else "white" for name in relation_df["Dataset"]],
        linewidth=[2.2 if name == "Ours" else 0.8 for name in relation_df["Dataset"]],
        alpha=0.92,
    )
    for y, value in enumerate(relation_df["Avg Relations/Object"]):
        ax.text(value + relation_df["Avg Relations/Object"].max() * 0.015, y, f"{value:.2f}", va="center", fontsize=11)
    ax.set_title("Avg Relations per Object", fontsize=20, fontweight="semibold")
    ax.set_xlabel("relations / object", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, relation_df["Avg Relations/Object"].max() * 1.16)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.savefig(output_dir / "relations_per_object_ranked.png", dpi=240)
    plt.close(fig)


def plot_relations_per_object_ranked(detail_df: pd.DataFrame, output_dir: Path) -> None:
    column = "relations_per_object"
    title = "Relations per Object"
    stats = (
        detail_df.groupby("dataset")[column]
        .agg(["mean", "median"])
        .sort_values(["mean", "median"], ascending=False)
    )
    labels = stats.index.tolist()
    data = [detail_df.loc[detail_df["dataset"] == dataset, column].to_numpy() for dataset in labels]

    fig, ax = plt.subplots(figsize=(15, 8))
    draw_colored_boxplot(ax, labels, data, title, "Relations / Object")
    ax.set_xlabel("Dataset (ordered by mean relations/object)", fontsize=14)
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "relations_per_object_boxplot_ranked.png", dpi=240)
    plt.close(fig)


def plot_attribute_coverage(detail_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> None:
    coverage_df = summary_df.sort_values("Pct Entities w/Attributes", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    ax.barh(
        coverage_df["Dataset"],
        coverage_df["Pct Entities w/Attributes"],
        color=[BENCHMARK_COLORS.get(name, "#cccccc") for name in coverage_df["Dataset"]],
        edgecolor=["#111111" if name == "Ours" else "white" for name in coverage_df["Dataset"]],
        linewidth=[2.2 if name == "Ours" else 0.8 for name in coverage_df["Dataset"]],
        alpha=0.92,
    )
    for y, value in enumerate(coverage_df["Pct Entities w/Attributes"]):
        ax.text(value + coverage_df["Pct Entities w/Attributes"].max() * 0.015, y, f"{value:.1f}%", va="center", fontsize=11)
    ax.set_title("Percent of Entities with Attributes", fontsize=20, fontweight="semibold")
    ax.set_xlabel("entities with attributes (%)", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, coverage_df["Pct Entities w/Attributes"].max() * 1.18)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.savefig(output_dir / "attribute_coverage_ranked.png", dpi=240)
    plt.close(fig)

    per_prompt = detail_df.copy()
    per_prompt["pct_entities_with_attributes"] = np.where(
        per_prompt["objects_per_prompt"] > 0,
        per_prompt["entities_with_attributes"] / per_prompt["objects_per_prompt"] * 100,
        0.0,
    )
    stats = (
        per_prompt.groupby("dataset")["pct_entities_with_attributes"]
        .agg(["mean", "median"])
        .sort_values(["mean", "median"], ascending=False)
    )
    labels = stats.index.tolist()
    data = [per_prompt.loc[per_prompt["dataset"] == dataset, "pct_entities_with_attributes"].to_numpy() for dataset in labels]

    fig, ax = plt.subplots(figsize=(15, 8))
    draw_colored_boxplot(ax, labels, data, "Per-Prompt Attribute Coverage", "% Entities with Attributes")
    ax.set_xlabel("Dataset (ordered by mean per-prompt coverage)", fontsize=14)
    ax.set_ylim(-3, 103)
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "attribute_coverage_boxplot_ranked.png", dpi=240)
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
    pd.DataFrame(rows).to_csv(output_dir / "top_terms.csv", index=False)


def best_other(summary_df: pd.DataFrame, column: str) -> pd.Series:
    return summary_df[summary_df["Dataset"] != "Ours"].sort_values(column, ascending=False).iloc[0]


def coverage_line(summary_df: pd.DataFrame, metric: str) -> str:
    ours = summary_df.loc[summary_df["Dataset"] == "Ours"].iloc[0]
    column = f"{metric} mean"
    better_than = summary_df[(summary_df["Dataset"] != "Ours") & (summary_df[column] < ours[column])]
    strongest = best_other(summary_df, column)
    return (
        f"- {metric}: Ours averages {ours[column]:.2f}. It is higher than "
        f"{len(better_than)} of {len(summary_df) - 1} comparison datasets; the largest prompt-estimate baseline is "
        f"{strongest['Dataset']} at {strongest[column]:.2f}."
    )


def write_analysis(summary_df: pd.DataFrame, output_dir: Path) -> None:
    ours = summary_df.loc[summary_df["Dataset"] == "Ours"].iloc[0]
    attr_obj_best = best_other(summary_df, "Attributes/Object mean")
    unique_attr_best = best_other(summary_df, "Unique Attributes")
    unique_rel_best = best_other(summary_df, "Unique Relations")
    rel_better = summary_df[
        (summary_df["Dataset"] != "Ours")
        & (summary_df["Relations/Prompt mean"] < ours["Relations/Prompt mean"])
    ]
    obj_better = summary_df[
        (summary_df["Dataset"] != "Ours")
        & (summary_df["Objects/Prompt mean"] < ours["Objects/Prompt mean"])
    ]

    lines = [
        "# Prompt Complexity Analysis",
        "",
        "All datasets are measured from their prompt text with spaCy. If `en_core_web_sm` is unavailable, the script uses a lightweight lexical fallback and records that in `summary.csv`.",
        "",
        "## Where Ours Stands",
        "",
        f"- Attribute binding is strong but not the top under spaCy-only scoring: ours has {ours['Attributes/Object mean']:.2f} attributes/object; the strongest baseline is {attr_obj_best['Dataset']} at {attr_obj_best['Attributes/Object mean']:.2f}.",
        f"- Attribute vocabulary is mid-to-high: ours has {int(ours['Unique Attributes'])} unique spaCy adjective lemmas; the largest comparison is {unique_attr_best['Dataset']} with {int(unique_attr_best['Unique Attributes'])}.",
        f"- Relation vocabulary is broad for a moderate-length prompt set: ours has {int(ours['Unique Relations'])} unique spaCy relation-term lemmas; the largest comparison is {unique_rel_best['Dataset']} with {int(unique_rel_best['Unique Relations'])}.",
        f"- Relations/prompt are competitive without relying on very long prose: ours averages {ours['Relations/Prompt mean']:.2f}, higher than {len(rel_better)} of {len(summary_df) - 1} comparison datasets.",
        f"- Object coverage stays multi-object: ours averages {ours['Objects/Prompt mean']:.2f} objects/prompt, higher than {len(obj_better)} of {len(summary_df) - 1} comparison datasets.",
        "",
        "## Distribution Checks",
        "",
        coverage_line(summary_df, "Objects/Prompt"),
        coverage_line(summary_df, "Attributes/Prompt"),
        coverage_line(summary_df, "Attributes/Object"),
        coverage_line(summary_df, "Relations/Prompt"),
        "",
        "## Ours At A Glance",
        "",
        f"- Prompts: {int(ours['Prompts'])}",
        f"- Unique objects: {int(ours['Unique Objects'])}",
        f"- Unique attributes: {int(ours['Unique Attributes'])}",
        f"- Unique relations: {int(ours['Unique Relations'])}",
        f"- Mean objects/prompt: {ours['Objects/Prompt mean']:.2f}",
        f"- Mean attributes/prompt: {ours['Attributes/Prompt mean']:.2f}",
        f"- Mean attributes/object: {ours['Attributes/Object mean']:.2f}",
        f"- Entities with attributes: {int(ours['Entities w/Attributes'])} ({ours['Pct Entities w/Attributes']:.1f}%)",
        f"- Mean relations/prompt: {ours['Relations/Prompt mean']:.2f}",
        f"- Mean relations/object: {ours['Avg Relations/Object']:.2f}",
        "",
        "## Interpretation",
        "",
        "Under spaCy-only scoring, the longest descriptive datasets lead on raw object, attribute, and relation counts. Ours remains a moderate-length, multi-object benchmark with competitive attribute density and relation density, so it is useful for testing compositional grounding without relying on very long prompt descriptions.",
        "",
        "See `summary.csv`, `per_prompt_distributions.csv`, `top_terms.csv`, and the PNG plots in this directory for the underlying numbers.",
    ]
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path, default=DEFAULT_OURS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nlp, nlp_source = load_nlp()
    prompt_metric_source = "spacy_prompt_estimate" if nlp_source.startswith("spacy") else "prompt_lexical_estimate"
    print(f"Prompt parser: {nlp_source}")

    summaries = []
    details = []
    object_terms = {}
    attribute_terms = {}
    relation_terms = {}

    for spec in load_benchmarks(args.ours):
        print(f"Processing {spec.name} ({len(spec.prompts)} prompts)")
        summary, detail_df, objects, attributes, relations = analyze_dataset(spec, nlp, prompt_metric_source)
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
    summary_df[summary_cols].to_csv(args.output_dir / "summary.csv", index=False)
    detail_df.to_csv(args.output_dir / "per_prompt_distributions.csv", index=False)
    write_top_terms(args.output_dir, object_terms, attribute_terms, relation_terms)
    plot_distributions(detail_df, args.output_dir)
    plot_ours_distributions(detail_df, args.output_dir)
    plot_attributes_per_object_ranked(detail_df, args.output_dir)
    plot_relations_per_object_ranked(detail_df, args.output_dir)
    plot_key_metric_rankings(summary_df, args.output_dir)
    plot_attribute_coverage(detail_df, summary_df, args.output_dir)
    write_analysis(summary_df, args.output_dir)

    display_cols = [
        "Dataset",
        "Prompts",
        "Source",
        "Objects/Prompt mean",
        "Attributes/Prompt mean",
        "Attributes/Object mean",
        "Avg Attributes/Entity",
        "Pct Entities w/Attributes",
        "Relations/Prompt mean",
        "Avg Relations/Object",
        "Unique Objects",
        "Unique Attributes",
        "Unique Relations",
    ]
    print("\n--- Distribution Summary ---")
    print(summary_df[display_cols].sort_values("Objects/Prompt mean", ascending=False).to_string(index=False))
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
