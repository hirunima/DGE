#!/usr/bin/env python3
"""
Interactive viewer for prompt-aligned image grids across baselines.
"""

import argparse
import base64
import html
import json
import os
import textwrap
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
    os.environ["MPLBACKEND"] = "Agg"

from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MODEL_DISPLAY_NAMES = {
    "flux": "FLUX.1 [dev]",
    "qwen-image": "Qwen-Image",
    "sd15": "Stable Diffusion v1.5",
    "sdxl": "Stable Diffusion XL 1.0",
    "z-image": "Z-Image-Turbo",
    "bagel": "Bagel",
}


def load_prompts(path: Path) -> List[dict]:
    with path.open() as f:
        return json.load(f)


def load_prompt_items(path: Path) -> Tuple[List[Tuple[str, str]], bool]:
    data = load_prompts(path)
    if isinstance(data, dict):
        items = sorted(data.items(), key=lambda item: int(item[0]))
        return [(k, v["prompt"] if isinstance(v, dict) else v) for k, v in items], True
    return [(str(i), entry["prompt"]) for i, entry in enumerate(data)], False


def load_question_counts(path: Path) -> Dict[str, Dict[str, float]]:
    import csv

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


def load_survey_pair_info(images_dir: Path) -> Dict[str, Dict[str, str]]:
    image1_dir = images_dir / "survey_samples" / "image1"
    image2_dir = images_dir / "survey_samples" / "image2"
    models1_path = image1_dir / "models.json"
    models2_path = image2_dir / "models.json"
    models1 = json.load(models1_path.open()) if models1_path.exists() else {}
    models2 = json.load(models2_path.open()) if models2_path.exists() else {}
    pairs: Dict[str, Dict[str, str]] = {}
    for filename, model in models1.items():
        idx = filename.split("-", 1)[0].zfill(4)
        pairs.setdefault(idx, {})["model_1"] = model
        pairs[idx]["image1_path"] = os.path.abspath(str(image1_dir / filename))
    for filename, model in models2.items():
        idx = filename.split("-", 1)[0].zfill(4)
        pairs.setdefault(idx, {})["model_2"] = model
        pairs[idx]["image2_path"] = os.path.abspath(str(image2_dir / filename))
    return pairs


def list_models(images_dir: Path) -> List[str]:
    models = []
    for entry in sorted(images_dir.iterdir()):
        if entry.is_dir() and entry.name != "survey_samples":
            models.append(entry.name)
    return models


def load_eval_scores(eval_dir: Path) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    for path in eval_dir.glob("*_eval_summary.json"):
        model = path.name.replace("_eval_summary.json", "")
        with path.open() as f:
            data = json.load(f)
        model_scores = {}
        for idx, entry in data.get("image_scores", {}).items():
            if "average" in entry:
                model_scores[str(int(idx))] = float(entry["average"])
        scores[model] = model_scores
    return scores


def load_jsonl_image_scores(path: Optional[Path]) -> Dict[str, float]:
    if not path or not path.exists():
        return {}
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


def load_vqa_scores(path: Optional[Path]) -> Dict[str, float]:
    return load_jsonl_image_scores(path)


def load_dsg_scores(path: Optional[Path]) -> Dict[str, float]:
    if not path or not path.exists():
        return {}
    scores: Dict[str, float] = {}
    with path.open() as f:
        data = json.load(f)
    for entry in data:
        image_path = entry.get("image_path")
        score = entry.get("final_dsg_score")
        if not image_path or score is None:
            continue
        scores[os.path.abspath(image_path)] = float(score)
    return scores


def image_path(images_dir: Path, model: str, idx: int, generation: int) -> Path:
    return images_dir / model / f"{idx:04d}-{generation}.png"


def load_image_or_none(path: Path) -> Optional[Image.Image]:
    if not path.exists():
        return None
    return Image.open(path).convert("RGB")


def score_rows_present(
    eval_scores: Dict[str, Dict[str, float]],
    dsg_scores: Dict[str, float],
    vqa_scores: Dict[str, float],
    pickscore_scores: Dict[str, float],
    imagereward_scores: Dict[str, float],
    human_prefs: Dict[str, Dict[str, float]],
) -> List[str]:
    rows = []
    if eval_scores:
        rows.append("Eval score")
    if pickscore_scores:
        rows.append("PickScore")
    if dsg_scores:
        rows.append("DSG score")
    if vqa_scores:
        rows.append("VQA score")
    if imagereward_scores:
        rows.append("ImageReward")
    if human_prefs:
        rows.append("Human Survey Preference")
    return rows


def format_score(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.2f}"


def outcome_status(
    score1: Optional[float],
    score2: Optional[float],
    human1: Optional[float],
    human2: Optional[float],
    eps: float = 1e-12,
) -> str:
    if score1 is None or score2 is None or human1 is None or human2 is None:
        return "unknown"
    human_delta = human1 - human2
    score_delta = score1 - score2
    if abs(human_delta) < eps:
        return "survey tie"
    if abs(score_delta) < eps:
        return "tie"
    return "right" if (human_delta > 0) == (score_delta > 0) else "wrong"


def label_with_outcome(
    label: str,
    score1: Optional[float],
    score2: Optional[float],
    human1: Optional[float],
    human2: Optional[float],
) -> str:
    if label == "Human Survey Preference":
        return label
    return f"{label} ({outcome_status(score1, score2, human1, human2)})"


def pair_scores_for_row(
    row_name: str,
    idx: int,
    eval_scores: Dict[str, Dict[str, float]],
    dsg_scores: Dict[str, float],
    vqa_scores: Dict[str, float],
    pickscore_scores: Dict[str, float],
    imagereward_scores: Dict[str, float],
    human_prefs: Dict[str, Dict[str, float]],
    survey_pair_info: Dict[str, Dict[str, str]],
) -> Tuple[Optional[float], Optional[float]]:
    idx_str = f"{idx:04d}"
    pair_info = survey_pair_info.get(idx_str, {})
    if row_name == "Eval score":
        return (
            eval_scores.get(pair_info.get("model_1", ""), {}).get(str(idx)),
            eval_scores.get(pair_info.get("model_2", ""), {}).get(str(idx)),
        )
    if row_name == "Human Survey Preference":
        human1 = human_prefs.get(idx_str, {}).get("preference")
        return human1, None if human1 is None else 1.0 - human1
    if row_name == "DSG score":
        scores = dsg_scores
    elif row_name == "VQA score":
        scores = vqa_scores
    elif row_name == "PickScore":
        scores = pickscore_scores
    else:
        scores = imagereward_scores
    return (
        scores.get(pair_info.get("image1_path", "")),
        scores.get(pair_info.get("image2_path", "")),
    )


def display_name(model_key: Optional[str]) -> str:
    if not model_key:
        return ""
    return MODEL_DISPLAY_NAMES.get(model_key, model_key)


def encode_image_base64(path: Path, size: Tuple[int, int]) -> str:
    with Image.open(path) as img:
        image = img.convert("RGB").resize(size)
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_svg(
    prompt: str,
    idx_str: str,
    model1_label: str,
    model2_label: str,
    image1_b64: str,
    image2_b64: str,
    eval1: Optional[float],
    eval2: Optional[float],
    dsg1: Optional[float],
    dsg2: Optional[float],
    vqa1: Optional[float],
    vqa2: Optional[float],
    pickscore1: Optional[float],
    pickscore2: Optional[float],
    imagereward1: Optional[float],
    imagereward2: Optional[float],
    human1: Optional[float],
    human2: Optional[float],
) -> str:
    words = prompt.split()
    lines = [" ".join(words[i : i + 5]) for i in range(0, len(words), 5)]
    max_lines = 5
    lines = lines[:max_lines]
    prompt_lines = "".join(
        f'<tspan x="150" dy="{18 if i > 0 else 0}">{html.escape(line)}</tspan>'
        for i, line in enumerate(lines)
    )
    eval_label = label_with_outcome("DGE-FineEval", eval1, eval2, human1, human2)
    pickscore_label = label_with_outcome("PickScore", pickscore1, pickscore2, human1, human2)
    dsg_label = label_with_outcome("DSG score", dsg1, dsg2, human1, human2)
    vqa_label = label_with_outcome("VQA score", vqa1, vqa2, human1, human2)
    imagereward_label = label_with_outcome("ImageReward", imagereward1, imagereward2, human1, human2)
    return f"""<svg width="1000" height="530" viewBox="0 0 1000 530" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#f0f0f0" stroke-width="1"/>
    </pattern>
    <linearGradient id="blueGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#d0eaff;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#eef7ff;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="purpleGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#e6dfff;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#f8f6ff;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="greenGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#d2f8d2;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#f2fff2;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="orangeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#ffe4bc;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#fff6e9;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="tealGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#d5f4f0;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#f1fffd;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="pinkGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#ffddec;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#fff5fa;stop-opacity:1" />
    </linearGradient>
  </defs>
  
  <rect width="1000" height="530" fill="#ffffff" />

  <text x="150" y="40" font-family="Segoe UI, Helvetica, Arial" font-size="18" font-weight="bold" text-anchor="middle" fill="#333">Text Prompt</text>
  <text x="500" y="45" font-family="Segoe UI, Helvetica, Arial" font-size="18" font-weight="bold" text-anchor="middle" fill="#333">{html.escape(model1_label)}</text>
  <text x="850" y="45" font-family="Segoe UI, Helvetica, Arial" font-size="18" font-weight="bold" text-anchor="middle" fill="#333">{html.escape(model2_label)}</text>
  
  <rect x="20" y="70" width="260" height="110" rx="12" fill="#ebebeb" stroke="#999" stroke-width="1"/>
  <text x="150" y="95" font-family="Segoe UI, Arial" font-size="13" text-anchor="middle" fill="#444">
    {prompt_lines}
  </text>
  
  <rect x="430" y="70" width="140" height="110" rx="8" fill="#fff" stroke="#999" stroke-width="1" />
  <image x="430" y="70" width="140" height="110" href="data:image/png;base64,{image1_b64}" />
  
  <rect x="780" y="70" width="140" height="110" rx="8" fill="#fff" stroke="#999" stroke-width="1" />
  <image x="780" y="70" width="140" height="110" href="data:image/png;base64,{image2_b64}" />

  <g transform="translate(0, 40)">
    <rect x="20" y="160" width="260" height="45" rx="10" fill="url(#blueGrad)" />
    <text x="150" y="188" font-family="Segoe UI, Arial" font-size="14" text-anchor="middle" fill="#222">{html.escape(eval_label)}</text>
    <rect x="360" y="160" width="280" height="45" rx="10" fill="url(#blueGrad)" />
    <text x="500" y="188" font-family="Segoe UI, Arial" font-size="16" font-weight="bold" text-anchor="middle">{format_score(eval1)}</text>
    <rect x="710" y="160" width="280" height="45" rx="10" fill="url(#blueGrad)" />
    <text x="850" y="188" font-family="Segoe UI, Arial" font-size="16" font-weight="bold" text-anchor="middle">{format_score(eval2)}</text>

    <rect x="20" y="215" width="260" height="45" rx="10" fill="url(#tealGrad)" />
    <text x="150" y="243" font-family="Segoe UI, Arial" font-size="14" text-anchor="middle" fill="#222">{html.escape(pickscore_label)}</text>
    <rect x="360" y="215" width="280" height="45" rx="10" fill="url(#tealGrad)" />
    <text x="500" y="243" font-family="Segoe UI, Arial" font-size="16" font-weight="bold" text-anchor="middle">{format_score(pickscore1)}</text>
    <rect x="710" y="215" width="280" height="45" rx="10" fill="url(#tealGrad)" />
    <text x="850" y="243" font-family="Segoe UI, Arial" font-size="16" font-weight="bold" text-anchor="middle">{format_score(pickscore2)}</text>

    <rect x="20" y="270" width="260" height="45" rx="10" fill="url(#purpleGrad)" />
    <text x="150" y="298" font-family="Segoe UI, Arial" font-size="14" text-anchor="middle" fill="#222">{html.escape(dsg_label)}</text>
    <rect x="360" y="270" width="280" height="45" rx="10" fill="url(#purpleGrad)" />
    <text x="500" y="298" font-family="Segoe UI, Arial" font-size="16" font-weight="bold" text-anchor="middle">{format_score(dsg1)}</text>
    <rect x="710" y="270" width="280" height="45" rx="10" fill="url(#purpleGrad)" />
    <text x="850" y="298" font-family="Segoe UI, Arial" font-size="16" font-weight="bold" text-anchor="middle">{format_score(dsg2)}</text>

    <rect x="20" y="325" width="260" height="45" rx="10" fill="url(#greenGrad)" />
    <text x="150" y="353" font-family="Segoe UI, Arial" font-size="14" text-anchor="middle" fill="#222">{html.escape(vqa_label)}</text>
    <rect x="360" y="325" width="280" height="45" rx="10" fill="url(#greenGrad)" />
    <text x="500" y="353" font-family="Segoe UI, Arial" font-size="16" font-weight="bold" text-anchor="middle">{format_score(vqa1)}</text>
    <rect x="710" y="325" width="280" height="45" rx="10" fill="url(#greenGrad)" />
    <text x="850" y="353" font-family="Segoe UI, Arial" font-size="16" font-weight="bold" text-anchor="middle">{format_score(vqa2)}</text>

    <rect x="20" y="380" width="260" height="45" rx="10" fill="url(#pinkGrad)" />
    <text x="150" y="408" font-family="Segoe UI, Arial" font-size="14" text-anchor="middle" fill="#222">{html.escape(imagereward_label)}</text>
    <rect x="360" y="380" width="280" height="45" rx="10" fill="url(#pinkGrad)" />
    <text x="500" y="408" font-family="Segoe UI, Arial" font-size="16" font-weight="bold" text-anchor="middle">{format_score(imagereward1)}</text>
    <rect x="710" y="380" width="280" height="45" rx="10" fill="url(#pinkGrad)" />
    <text x="850" y="408" font-family="Segoe UI, Arial" font-size="16" font-weight="bold" text-anchor="middle">{format_score(imagereward2)}</text>

    <rect x="20" y="435" width="260" height="45" rx="10" fill="url(#orangeGrad)" />
    <text x="150" y="463" font-family="Segoe UI, Arial" font-size="15" text-anchor="middle" fill="#222">Human Survey Preference</text>
    <rect x="360" y="435" width="280" height="45" rx="10" fill="url(#orangeGrad)" />
    <text x="500" y="463" font-family="Segoe UI, Arial" font-size="18" font-weight="bold" text-anchor="middle">{format_score(human1)}</text>
    <rect x="710" y="435" width="280" height="45" rx="10" fill="url(#orangeGrad)" />
    <text x="850" y="463" font-family="Segoe UI, Arial" font-size="16" font-weight="bold" text-anchor="middle">{format_score(human2)}</text>
  </g>
</svg>
"""


def render_prompt_grid(
    fig: plt.Figure,
    axes,
    prompt: str,
    idx: int,
    image_infos: List[Dict[str, Optional[str]]],
    eval_scores: Dict[str, Dict[str, float]],
    dsg_scores: Dict[str, float],
    vqa_scores: Dict[str, float],
    pickscore_scores: Dict[str, float],
    imagereward_scores: Dict[str, float],
    human_prefs: Dict[str, Dict[str, float]],
    survey_pair_info: Dict[str, Dict[str, str]],
    generation: int,
) -> None:
    score_rows = score_rows_present(
        eval_scores,
        dsg_scores,
        vqa_scores,
        pickscore_scores,
        imagereward_scores,
        human_prefs,
    )
    n_rows = 1 + len(score_rows)
    n_cols = 1 + len(image_infos)

    for row in axes:
        for ax in row:
            ax.clear()
            ax.set_axis_off()

    # Prompt cell
    ax_prompt = axes[0][0]
    ax_prompt.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor="#f2f2f2", edgecolor="none"))
    ax_prompt.set_xlim(0, 1)
    ax_prompt.set_ylim(0, 1)
    words = prompt.split()
    prompt_wrapped = "\n".join(
        " ".join(words[i : i + 5]) for i in range(0, len(words), 5)
    )
    word_count = len(words)
    prompt_font = 16 if word_count <= 12 else 14 if word_count <= 20 else 12
    ax_prompt.text(
        0.5,
        0.5,
        prompt_wrapped,
        ha="center",
        va="center",
        wrap=True,
        fontsize=prompt_font,
    )
    ax_prompt.set_title("Text Prompt", fontsize=14, pad=10)

    # Image columns
    for col, info in enumerate(image_infos, start=1):
        ax = axes[0][col]
        model = info.get("title") or "image"
        image = info.get("image")
        if image is None:
            ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor="#dddddd", edgecolor="none"))
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12)
        else:
            ax.imshow(image)
        ax.set_title(model, fontsize=14, pad=10)

    # Score rows
    for row_idx, row_name in enumerate(score_rows, start=1):
        ax_label = axes[row_idx][0]
        if row_name == "Eval score":
            bg = "#e8f4ff"
        elif row_name == "PickScore":
            bg = "#e2f6f2"
        elif row_name == "DSG score":
            bg = "#f4f0ff"
        elif row_name == "Human Survey Preference":
            bg = "#fff2cc"
        elif row_name == "ImageReward":
            bg = "#ffe9f2"
        else:
            bg = "#e6ffe6"
        row_score1, row_score2 = pair_scores_for_row(
            row_name,
            idx,
            eval_scores,
            dsg_scores,
            vqa_scores,
            pickscore_scores,
            imagereward_scores,
            human_prefs,
            survey_pair_info,
        )
        human1, human2 = pair_scores_for_row(
            "Human Survey Preference",
            idx,
            eval_scores,
            dsg_scores,
            vqa_scores,
            pickscore_scores,
            imagereward_scores,
            human_prefs,
            survey_pair_info,
        )
        row_label = label_with_outcome(row_name, row_score1, row_score2, human1, human2)
        ax_label.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor=bg, edgecolor="none"))
        ax_label.text(0.5, 0.5, row_label, ha="center", va="center", fontsize=11)

        for col, info in enumerate(image_infos, start=1):
            ax_cell = axes[row_idx][col]
            ax_cell.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor=bg, edgecolor="none"))
            if row_name == "Eval score":
                model_name = info.get("model")
                value = eval_scores.get(model_name or "", {}).get(str(idx))
            elif row_name == "DSG score":
                model_name = info.get("model")
                idx_str = f"{idx:04d}"
                pair_info = survey_pair_info.get(idx_str, {})
                if model_name == pair_info.get("model_1"):
                    value = dsg_scores.get(pair_info.get("image1_path", ""))
                elif model_name == pair_info.get("model_2"):
                    value = dsg_scores.get(pair_info.get("image2_path", ""))
                else:
                    value = None
            elif row_name == "Human Survey Preference":
                model_name = info.get("model")
                idx_str = f"{idx:04d}"
                pref_entry = human_prefs.get(idx_str)
                pair_models = survey_pair_info.get(idx_str, {})
                if pref_entry and model_name:
                    if model_name == pair_models.get("model_1"):
                        value = pref_entry.get("preference")
                    elif model_name == pair_models.get("model_2"):
                        value = 1.0 - pref_entry.get("preference", 0.5)
                    else:
                        value = None
                else:
                    value = None
            elif row_name == "VQA score":
                model_name = info.get("model")
                idx_str = f"{idx:04d}"
                pair_info = survey_pair_info.get(idx_str, {})
                if model_name == pair_info.get("model_1"):
                    value = vqa_scores.get(pair_info.get("image1_path", ""))
                elif model_name == pair_info.get("model_2"):
                    value = vqa_scores.get(pair_info.get("image2_path", ""))
                else:
                    value = None
            else:
                model_name = info.get("model")
                idx_str = f"{idx:04d}"
                pair_info = survey_pair_info.get(idx_str, {})
                row_scores = pickscore_scores if row_name == "PickScore" else imagereward_scores
                if model_name == pair_info.get("model_1"):
                    value = row_scores.get(pair_info.get("image1_path", ""))
                elif model_name == pair_info.get("model_2"):
                    value = row_scores.get(pair_info.get("image2_path", ""))
                else:
                    value = None
            ax_cell.text(0.5, 0.5, format_score(value), ha="center", va="center", fontsize=12)

    fig.suptitle(f"Prompt {idx:04d}", fontsize=12, y=0.98)
    fig.canvas.draw_idle()


def build_axes(fig: plt.Figure, n_rows: int, n_cols: int):
    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=[3] + [1] * (n_rows - 1))
    axes = []
    for r in range(n_rows):
        row_axes = []
        for c in range(n_cols):
            ax = fig.add_subplot(gs[r, c])
            row_axes.append(ax)
        axes.append(row_axes)
    return axes


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive prompt grid viewer.")
    parser.add_argument("--prompts", type=Path, default=Path("data/raw/qwen8b_t2i_prompts_aug_v1.json"))
    parser.add_argument("--survey_prompts", type=Path, default=Path("data/images/survey_samples/prompts.json"))
    parser.add_argument("--images_dir", type=Path, default=Path("data/images"))
    parser.add_argument("--eval_dir", type=Path, default=Path("data/raw/eval_v1"))
    parser.add_argument("--dsg_scores", type=Path, default=Path("../../DSG/evaluation_results_dge.json"))
    parser.add_argument("--vqa_scores", type=Path, default=Path("../../t2v_metrics/results/survey_samples_vqa_scores.jsonl"))
    parser.add_argument("--pickscore_scores", type=Path, default=Path("reports/baselines/survey_samples_pickscore.jsonl"))
    parser.add_argument("--imagereward_scores", type=Path, default=Path("reports/baselines/survey_samples_imagereward.jsonl"))
    parser.add_argument("--human_counts", type=Path, default=Path("data/images/survey_samples/question_counts.csv"))
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model list.")
    parser.add_argument("--exclude_models", type=str, default="bagel,survey_samples")
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Number of prompts to render in headless mode.")
    parser.add_argument("--headless", action="store_true", help="Render to files instead of opening a window.")
    parser.add_argument("--no_ask", action="store_true", help="Do not prompt in headless mode.")
    parser.add_argument("--only_ids", type=str, default=None, help="Comma-separated prompt ids to include (e.g., 0003,2790).")
    parser.add_argument("--only_ids_file", type=Path, default=None, help="Text file with one prompt id per line.")
    parser.add_argument("--save_single", action="store_true", help="Overwrite a single PNG instead of per-id files.")
    parser.add_argument("--figsize", type=str, default="15.5,5.0")
    parser.add_argument("--dpi", type=int, default=130)
    args = parser.parse_args()

    prompt_items, is_survey = load_prompt_items(args.survey_prompts)
    if not prompt_items:
        raise SystemExit(f"No prompts found at {args.survey_prompts}.")
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        excluded = {m.strip() for m in args.exclude_models.split(",") if m.strip()}
        models = [m for m in list_models(args.images_dir) if m not in excluded]

    eval_scores = load_eval_scores(args.eval_dir) if args.eval_dir.exists() else {}
    dsg_scores = load_dsg_scores(args.dsg_scores)
    vqa_scores = load_vqa_scores(args.vqa_scores)
    pickscore_scores = load_jsonl_image_scores(args.pickscore_scores)
    imagereward_scores = load_jsonl_image_scores(args.imagereward_scores)
    human_prefs = load_question_counts(args.human_counts)
    survey_pair_info = load_survey_pair_info(args.images_dir)

    if args.only_ids or args.only_ids_file:
        ids = set()
        if args.only_ids:
            ids.update(i.strip().zfill(4) for i in args.only_ids.split(",") if i.strip())
        if args.only_ids_file and args.only_ids_file.exists():
            ids.update(i.strip().zfill(4) for i in args.only_ids_file.read_text().splitlines() if i.strip())
        prompt_items = [(idx_str, prompt) for idx_str, prompt in prompt_items if idx_str in ids]
        if not prompt_items:
            raise SystemExit("No prompts left after filtering by only_ids.")
    idx = max(0, min(args.start_idx, len(prompt_items) - 1))

    score_rows = score_rows_present(
        eval_scores,
        dsg_scores,
        vqa_scores,
        pickscore_scores,
        imagereward_scores,
        human_prefs,
    )
    n_rows = 1 + len(score_rows)
    n_cols = 3

    figsize = tuple(float(v) for v in args.figsize.split(","))
    fig = plt.figure(figsize=figsize, dpi=args.dpi)
    axes = build_axes(fig, n_rows, n_cols)

    def load_infos_for(idx_value: int, idx_str: str) -> List[Dict[str, Optional[str]]]:
        infos: List[Dict[str, Optional[str]]] = []
        pair_info = survey_pair_info.get(idx_str, {})
        model1 = pair_info.get("model_1")
        model2 = pair_info.get("model_2")
        path1 = pair_info.get("image1_path")
        path2 = pair_info.get("image2_path")
        if model1 and path1:
            infos.append(
                {
                    "title": model1,
                    "model": model1,
                    "filename": Path(path1).name,
                    "path": os.path.abspath(path1),
                    "image": load_image_or_none(Path(path1)),
                }
            )
        if model2 and path2:
            infos.append(
                {
                    "title": model2,
                    "model": model2,
                    "filename": Path(path2).name,
                    "path": os.path.abspath(path2),
                    "image": load_image_or_none(Path(path2)),
                }
            )
        return infos

    headless = args.headless or matplotlib.get_backend().lower() == "agg"
    if headless:
        if not args.save_dir:
            raise SystemExit("Headless mode requires --save_dir.")
        args.save_dir.mkdir(parents=True, exist_ok=True)
        end_idx = len(prompt_items) if args.limit is None else min(len(prompt_items), idx + args.limit)
        ask = not args.no_ask
        for current in range(idx, end_idx):
            idx_str, prompt = prompt_items[current]
            idx_value = int(idx_str)
            infos = load_infos_for(idx_value, idx_str)
            render_prompt_grid(
                fig,
                axes,
                prompt,
                idx_value,
                infos,
                eval_scores,
                dsg_scores,
                vqa_scores,
                pickscore_scores,
                imagereward_scores,
                human_prefs,
                survey_pair_info,
                args.generation,
            )
            pair_info = survey_pair_info.get(idx_str, {})
            image1_path = pair_info.get("image1_path")
            image2_path = pair_info.get("image2_path")
            model1_key = pair_info.get("model_1", "model_1")
            model2_key = pair_info.get("model_2", "model_2")
            model1_label = display_name(model1_key)
            model2_label = display_name(model2_key)
            if not image1_path or not image2_path:
                raise SystemExit(f"Missing survey image paths for {idx_str}.")
            image1_b64 = encode_image_base64(Path(image1_path), (140, 110))
            image2_b64 = encode_image_base64(Path(image2_path), (140, 110))
            eval1 = eval_scores.get(model1_key, {}).get(str(int(idx_str)))
            eval2 = eval_scores.get(model2_key, {}).get(str(int(idx_str)))
            dsg1 = dsg_scores.get(image1_path)
            dsg2 = dsg_scores.get(image2_path)
            vqa1 = vqa_scores.get(image1_path)
            vqa2 = vqa_scores.get(image2_path)
            pickscore1 = pickscore_scores.get(image1_path)
            pickscore2 = pickscore_scores.get(image2_path)
            imagereward1 = imagereward_scores.get(image1_path)
            imagereward2 = imagereward_scores.get(image2_path)
            human_entry = human_prefs.get(idx_str, {})
            human1 = human_entry.get("preference")
            human2 = None if human1 is None else 1.0 - human1
            svg = build_svg(
                prompt=prompt,
                idx_str=idx_str,
                model1_label=model1_label,
                model2_label=model2_label,
                image1_b64=image1_b64,
                image2_b64=image2_b64,
                eval1=eval1,
                eval2=eval2,
                dsg1=dsg1,
                dsg2=dsg2,
                vqa1=vqa1,
                vqa2=vqa2,
                pickscore1=pickscore1,
                pickscore2=pickscore2,
                imagereward1=imagereward1,
                imagereward2=imagereward2,
                human1=human1,
                human2=human2,
            )
            if args.save_single:
                out_path = args.save_dir / "prompt_grid.svg"
            else:
                out_path = args.save_dir / f"prompt_{int(idx_str):04d}.svg"
            out_path.write_text(svg)
            if ask:
                resp = input("Continue? [y/N] ").strip().lower()
                if resp not in {"y", "yes"}:
                    break
        return

    idx_str, prompt = prompt_items[idx]
    infos = load_infos_for(int(idx_str), idx_str)
    render_prompt_grid(
        fig,
        axes,
        prompt,
        int(idx_str),
        infos,
        eval_scores,
        dsg_scores,
        vqa_scores,
        pickscore_scores,
        imagereward_scores,
        human_prefs,
        survey_pair_info,
        args.generation,
    )

    def on_key(event):
        nonlocal idx, infos
        if event.key in {"right", "n"}:
            idx = (idx + 1) % len(prompt_items)
        elif event.key in {"left", "p"}:
            idx = (idx - 1) % len(prompt_items)
        elif event.key == "s" and args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            current_idx_str = prompt_items[idx][0]
            out_path = args.save_dir / f"prompt_{int(current_idx_str):04d}.png"
            fig.savefig(out_path, bbox_inches="tight")
            return
        elif event.key in {"q", "escape"}:
            plt.close(fig)
            return
        else:
            return

        idx_str, prompt = prompt_items[idx]
        infos = load_infos_for(int(idx_str), idx_str)
        render_prompt_grid(
            fig,
            axes,
            prompt,
            int(idx_str),
            infos,
            eval_scores,
            dsg_scores,
            vqa_scores,
            pickscore_scores,
            imagereward_scores,
            human_prefs,
            survey_pair_info,
            args.generation,
        )

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
