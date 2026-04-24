#!/usr/bin/env python3
"""
Compute average human preference per survey pair and compare against eval scores.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_question_counts(path: Path) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Dict[str, float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row["id"].zfill(4)
            m1 = int(row["model_1"])
            m2 = int(row["model_2"])
            tie = int(row["tie"])
            total = m1 + m2 + tie
            preference = (m1 + 0.5 * tie) / total if total else 0.5
            counts[idx] = {
                "model_1": m1,
                "model_2": m2,
                "tie": tie,
                "total": total,
                "preference": preference,
            }
    return counts


def load_models(path: Path) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    with path.open() as f:
        filename_to_model = json.load(f)
    id_to_files: Dict[str, List[str]] = {}
    for filename in filename_to_model.keys():
        idx = filename.split("-", 1)[0].zfill(4)
        id_to_files.setdefault(idx, []).append(filename)
    return id_to_files, filename_to_model


def load_eval_summaries(eval_dir: Path) -> Dict[str, Dict[str, float]]:
    summaries: Dict[str, Dict[str, float]] = {}
    for path in eval_dir.glob("*_eval_summary.json"):
        model = path.name.replace("_eval_summary.json", "")
        with path.open() as f:
            data = json.load(f)
        scores = {}
        for idx, entry in data.get("image_scores", {}).items():
            if "average" in entry:
                scores[str(int(idx))] = float(entry["average"])
        summaries[model] = scores
    return summaries


def load_dsg_scores(path: Path) -> Dict[str, float]:
    with path.open() as f:
        data = json.load(f)
    scores: Dict[str, float] = {}
    for entry in data:
        image_path = entry.get("image_path")
        if not image_path:
            continue
        score = entry.get("final_dsg_score")
        if score is None:
            continue
        scores[os.path.abspath(image_path)] = float(score)
    return scores


def load_vqa_scores(path: Path) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            image_path = entry.get("image")
            score = entry.get("score")
            if not image_path or score is None:
                continue
            scores[os.path.abspath(image_path)] = float(score)
    return scores


def kendall_tau_b(xs: List[float], ys: List[float], eps: float = 1e-12) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if abs(dx) < eps and abs(dy) < eps:
                continue
            if abs(dx) < eps:
                ties_x += 1
                continue
            if abs(dy) < eps:
                ties_y += 1
                continue
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return None
    return (concordant - discordant) / denom


def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def rankdata(values: List[float], eps: float = 1e-12) -> List[float]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and abs(indexed[j][1] - indexed[i][1]) < eps:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    rx = rankdata(xs)
    ry = rankdata(ys)
    return pearson(rx, ry)


def pairwise_accuracy(xs: List[float], ys: List[float], eps: float = 1e-12) -> Tuple[Optional[float], int, int]:
    correct = 0
    total = 0
    ties = 0
    for x, y in zip(xs, ys):
        dx = x - 0.5
        dy = y - 0.5
        if abs(dx) < eps:
            ties += 1
            continue
        if abs(dy) < eps:
            total += 1
            continue
        total += 1
        if (dx > 0) == (dy > 0):
            correct += 1
    return (correct / total if total else None), total, ties


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare survey preferences with eval scores.")
    parser.add_argument("--counts_csv", type=Path, default=Path("data/images/survey_samples/question_counts.csv"))
    parser.add_argument("--models1", type=Path, default=Path("data/images/survey_samples/image1/models.json"))
    parser.add_argument("--models2", type=Path, default=Path("data/images/survey_samples/image2/models.json"))
    parser.add_argument("--eval_dir", type=Path, default=Path("data/raw/eval_v1"))
    parser.add_argument("--dsg_results", type=Path, default=Path("/fs/nexus-projects/scene_graph_sd/DSG/evaluation_results_dge.json"))
    parser.add_argument("--vqa_results", type=Path, default=Path("/fs/nexus-projects/scene_graph_sd/t2v_metrics/results/survey_samples_vqa_scores.jsonl"))
    parser.add_argument("--output_csv", type=Path, default=Path("data/images/survey_samples/pair_preferences.csv"))
    parser.add_argument("--metrics_json", type=Path, default=Path("reports/pair_metrics.json"))
    args = parser.parse_args()

    counts = load_question_counts(args.counts_csv)
    image1_map, image1_models = load_models(args.models1)
    image2_map, image2_models = load_models(args.models2)
    eval_summaries = load_eval_summaries(args.eval_dir)
    dsg_scores = load_dsg_scores(args.dsg_results)
    vqa_scores = load_vqa_scores(args.vqa_results)

    rows = []
    survey_prefs_eval = []
    eval_prefs = []
    survey_prefs_dsg = []
    dsg_prefs = []
    survey_prefs_vqa = []
    vqa_prefs = []
    missing_eval = 0
    missing_dsg = 0
    missing_vqa = 0

    for idx, entry in counts.items():
        image1_files = image1_map.get(idx, [])
        image2_files = image2_map.get(idx, [])
        if not image1_files or not image2_files:
            continue
        image1_file = image1_files[0]
        image2_file = image2_files[0]
        model1 = image1_models.get(image1_file)
        model2 = image2_models.get(image2_file)
        if not model1 or not model2:
            continue

        survey_pref = entry["preference"]

        eval_score1 = eval_summaries.get(model1, {}).get(str(int(idx)))
        eval_score2 = eval_summaries.get(model2, {}).get(str(int(idx)))
        if eval_score1 is None or eval_score2 is None:
            missing_eval += 1
            eval_pref = None
        else:
            denom = eval_score1 + eval_score2
            eval_pref = eval_score1 / denom if denom else 0.5

        image1_path = os.path.abspath(str(Path("data/images/survey_samples/image1") / image1_file))
        image2_path = os.path.abspath(str(Path("data/images/survey_samples/image2") / image2_file))
        dsg_score1 = dsg_scores.get(image1_path)
        dsg_score2 = dsg_scores.get(image2_path)
        if dsg_score1 is None or dsg_score2 is None:
            missing_dsg += 1
            dsg_pref = None
        else:
            denom = dsg_score1 + dsg_score2
            dsg_pref = dsg_score1 / denom if denom else 0.5

        vqa_score1 = vqa_scores.get(image1_path)
        vqa_score2 = vqa_scores.get(image2_path)
        if vqa_score1 is None or vqa_score2 is None:
            missing_vqa += 1
            vqa_pref = None
        else:
            denom = vqa_score1 + vqa_score2
            vqa_pref = vqa_score1 / denom if denom else 0.5

        rows.append(
            {
                "id": idx,
                "image1_file": image1_file,
                "image2_file": image2_file,
                "model_1": model1,
                "model_2": model2,
                "model_1_count": entry["model_1"],
                "model_2_count": entry["model_2"],
                "tie_count": entry["tie"],
                "survey_pref_model1": survey_pref,
                "eval_score_model1": eval_score1,
                "eval_score_model2": eval_score2,
                "eval_pref_model1": eval_pref,
                "dsg_score_model1": dsg_score1,
                "dsg_score_model2": dsg_score2,
                "dsg_pref_model1": dsg_pref,
                "vqa_score_model1": vqa_score1,
                "vqa_score_model2": vqa_score2,
                "vqa_pref_model1": vqa_pref,
            }
        )

        if eval_pref is not None:
            survey_prefs_eval.append(survey_pref)
            eval_prefs.append(eval_pref)
        if dsg_pref is not None:
            survey_prefs_dsg.append(survey_pref)
            dsg_prefs.append(dsg_pref)
        if vqa_pref is not None:
            survey_prefs_vqa.append(survey_pref)
            vqa_prefs.append(vqa_pref)

    if rows:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    metrics = {"eval_v1": None, "dsg": None}

    if eval_prefs:
        eval_pearson = pearson(survey_prefs_eval, eval_prefs)
        eval_spearman = spearman(survey_prefs_eval, eval_prefs)
        eval_kendall = kendall_tau_b(survey_prefs_eval, eval_prefs)
        eval_acc, eval_total, eval_ties = pairwise_accuracy(survey_prefs_eval, eval_prefs)
        metrics["eval_v1"] = {
            "pearson": eval_pearson,
            "spearman": eval_spearman,
            "kendall_tau_b": eval_kendall,
            "pairwise_accuracy": eval_acc,
            "n": eval_total,
            "survey_ties": eval_ties,
        }
        print("Eval v1 vs survey:")
        print(f"  pearson={eval_pearson}")
        print(f"  spearman={eval_spearman}")
        print(f"  kendall_tau_b={eval_kendall}")
        print(f"  pairwise_accuracy={eval_acc} (n={eval_total}, survey_ties={eval_ties})")
    else:
        print("Eval v1 vs survey: no overlapping pairs.")

    if dsg_prefs:
        dsg_pearson = pearson(survey_prefs_dsg, dsg_prefs)
        dsg_spearman = spearman(survey_prefs_dsg, dsg_prefs)
        dsg_kendall = kendall_tau_b(survey_prefs_dsg, dsg_prefs)
        dsg_acc, dsg_total, dsg_ties = pairwise_accuracy(survey_prefs_dsg, dsg_prefs)
        metrics["dsg"] = {
            "pearson": dsg_pearson,
            "spearman": dsg_spearman,
            "kendall_tau_b": dsg_kendall,
            "pairwise_accuracy": dsg_acc,
            "n": dsg_total,
            "survey_ties": dsg_ties,
        }
        print("DSG vs survey:")
        print(f"  pearson={dsg_pearson}")
        print(f"  spearman={dsg_spearman}")
        print(f"  kendall_tau_b={dsg_kendall}")
        print(f"  pairwise_accuracy={dsg_acc} (n={dsg_total}, survey_ties={dsg_ties})")
    else:
        print("DSG vs survey: no overlapping pairs.")

    if vqa_prefs:
        vqa_pearson = pearson(survey_prefs_vqa, vqa_prefs)
        vqa_spearman = spearman(survey_prefs_vqa, vqa_prefs)
        vqa_kendall = kendall_tau_b(survey_prefs_vqa, vqa_prefs)
        vqa_acc, vqa_total, vqa_ties = pairwise_accuracy(survey_prefs_vqa, vqa_prefs)
        metrics["vqa"] = {
            "pearson": vqa_pearson,
            "spearman": vqa_spearman,
            "kendall_tau_b": vqa_kendall,
            "pairwise_accuracy": vqa_acc,
            "n": vqa_total,
            "survey_ties": vqa_ties,
        }
        print("VQA vs survey:")
        print(f"  pearson={vqa_pearson}")
        print(f"  spearman={vqa_spearman}")
        print(f"  kendall_tau_b={vqa_kendall}")
        print(f"  pairwise_accuracy={vqa_acc} (n={vqa_total}, survey_ties={vqa_ties})")
    else:
        print("VQA vs survey: no overlapping pairs.")

    print(f"Missing eval_v1 pairs: {missing_eval}")
    print(f"Missing DSG pairs: {missing_dsg}")
    print(f"Missing VQA pairs: {missing_vqa}")
    if rows:
        print(f"Wrote per-pair data to {args.output_csv}")
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_json.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics to {args.metrics_json}")


if __name__ == "__main__":
    main()
