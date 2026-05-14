#!/usr/bin/env python3
"""Overlay detections from an ablation details JSON on an image."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def load_details(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def point_attributes(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        attr
        for attr in entry.get("st2_res", {}).get("attributes", [])
        if attr.get("points")
    ]


def detected_nodes(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        node
        for node in entry.get("st1_res", {}).get("nodes", [])
        if node.get("bbox")
    ]


def find_auto_example(details: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for entry in details:
        nodes = entry.get("st1_res", {}).get("nodes", [])
        attrs = entry.get("st2_res", {}).get("attributes", [])
        rels = entry.get("st3_res", {}).get("relations", [])
        point_attrs = point_attributes(entry)
        count_attrs = [
            attr
            for attr in attrs
            if attr.get("expected_count") is not None or "quantity:" in str(attr.get("attribute", ""))
        ]
        point_count = sum(len(attr.get("points", [])) for attr in point_attrs)
        if len(nodes) >= 4 and len(attrs) >= 3 and rels and count_attrs and point_count >= 2:
            candidates.append((point_count, len(nodes), len(attrs), entry))
    if not candidates:
        raise ValueError("No example found with objects, attributes, count attributes, relations, and points")
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def load_image1_models(path: Path) -> dict[str, str]:
    models: dict[str, str] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_file = row.get("image1_file")
            model = row.get("model_1")
            if image_file and model:
                models[image_file] = model
    return models


def resolve_image_path(
    image_id: str,
    image_path: Path | None,
    image_root: Path | None,
    survey_csv: Path | None,
) -> Path:
    if image_path:
        return image_path

    filename = f"{image_id}.png"
    model: str | None = None
    if survey_csv and survey_csv.exists():
        model = load_image1_models(survey_csv).get(filename)

    search_dirs: list[Path] = []
    if image_root:
        if model:
            search_dirs.append(image_root / model)
        search_dirs.append(image_root)

    for directory in search_dirs:
        for ext in IMAGE_EXTENSIONS:
            candidate = directory / f"{image_id}{ext}"
            if candidate.exists():
                return candidate

    searched = ", ".join(str(path) for path in search_dirs) or "no image directories"
    raise FileNotFoundError(f"Could not find image for {image_id}; searched {searched}")


def scale_point(point: dict[str, Any], width: int, height: int, coord_width: float, coord_height: float) -> tuple[float, float]:
    return float(point["x"]) * width / coord_width, float(point["y"]) * height / coord_height


def coordinate_space(entry: dict[str, Any], attrs: list[dict[str, Any]]) -> tuple[float, float]:
    points = [point for attr in attrs for point in attr.get("points", [])]
    bboxes = [node["bbox"] for node in detected_nodes(entry)]
    coord_width = max([1024.0] + [float(point["x"]) for point in points] + [float(bbox[2]) for bbox in bboxes])
    coord_height = max([1024.0] + [float(point["y"]) for point in points] + [float(bbox[3]) for bbox in bboxes])
    return coord_width, coord_height


def draw_points(
    draw: ImageDraw.ImageDraw,
    attrs: list[dict[str, Any]],
    width: int,
    height: int,
    coord_width: float,
    coord_height: float,
    radius: int,
) -> int:
    marker_radius = radius or max(8, int(min(width, height) * 0.014))
    point_count = 0

    for attr in attrs:
        for point in attr.get("points", []):
            x, y = scale_point(point, width, height, coord_width, coord_height)
            draw.ellipse(
                (x - marker_radius, y - marker_radius, x + marker_radius, y + marker_radius),
                fill="#ff2d55",
                outline="white",
                width=max(2, marker_radius // 4),
            )
            cross = marker_radius * 1.5
            draw.line((x - cross, y, x + cross, y), fill="white", width=max(1, marker_radius // 5))
            draw.line((x, y - cross, x, y + cross), fill="white", width=max(1, marker_radius // 5))
            point_count += 1
    return point_count


def draw_boxes(
    draw: ImageDraw.ImageDraw,
    nodes: list[dict[str, Any]],
    width: int,
    height: int,
    coord_width: float,
    coord_height: float,
) -> int:
    line_width = max(10, int(min(width, height) * 0.006))
    for node in nodes:
        x1, y1, x2, y2 = [float(value) for value in node["bbox"]]
        box = (
            x1 * width / coord_width,
            y1 * height / coord_height,
            x2 * width / coord_width,
            y2 * height / coord_height,
        )
        draw.rectangle(box, outline="#00a6ff", width=line_width)
    return len(nodes)


def draw_overlay(image: Image.Image, entry: dict[str, Any], overlay: str, radius: int) -> tuple[Image.Image, int, int]:
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    width, height = output.size
    attrs = point_attributes(entry)
    nodes = detected_nodes(entry)
    coord_width, coord_height = coordinate_space(entry, attrs)
    point_count = 0
    box_count = 0

    if overlay in {"boxes", "both"}:
        box_count = draw_boxes(draw, nodes, width, height, coord_width, coord_height)
    if overlay in {"points", "both"}:
        point_count = draw_points(draw, attrs, width, height, coord_width, coord_height, radius)
    return output, point_count, box_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--details",
        type=Path,
        default=Path("reports/ablation/validation/permutations_img1/V1-V2-V3_details.json"),
        help="Ablation details JSON containing st2_res attribute point detections.",
    )
    parser.add_argument("--image-id", help="Image id to render. If omitted, a matching example is selected.")
    parser.add_argument("--image-path", type=Path, help="Exact source image path. Overrides image-root lookup.")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("/fs/nexus-projects/scene_graph_sd/DGE/DGE-T2I-og/data/images/split/validation"),
        help="Root containing model subdirectories such as flux, sdxl, and z-image.",
    )
    parser.add_argument(
        "--survey-csv",
        type=Path,
        default=Path("data/images/survey_samples/pair_preferences.csv"),
        help="CSV used to map image1 ids to their model directory.",
    )
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    parser.add_argument(
        "--overlay",
        choices=("points", "boxes", "both"),
        default="points",
        help="Detection overlay to draw. Default: points.",
    )
    parser.add_argument("--radius", type=int, default=0, help="Point marker radius in pixels. Default scales to image size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    details = load_details(args.details)
    if args.image_id:
        entry = next((item for item in details if item.get("image_id") == args.image_id), None)
        if entry is None:
            raise ValueError(f"Image id {args.image_id!r} not found in {args.details}")
    else:
        entry = find_auto_example(details)

    attrs = point_attributes(entry)
    nodes = detected_nodes(entry)
    if args.overlay in {"points", "both"} and not attrs:
        raise ValueError(f"Image id {entry['image_id']} has no detected points")
    if args.overlay in {"boxes", "both"} and not nodes:
        raise ValueError(f"Image id {entry['image_id']} has no detected bounding boxes")

    image_path = resolve_image_path(entry["image_id"], args.image_path, args.image_root, args.survey_csv)
    image = Image.open(image_path)
    output, point_count, box_count = draw_overlay(image, entry, args.overlay, args.radius)
    output_path = args.output or args.details.with_name(f"{entry['image_id']}_{args.overlay}_overlay.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)

    print(f"image_id: {entry['image_id']}")
    print(f"source: {image_path}")
    print(f"output: {output_path}")
    print(f"points: {point_count}")
    print(f"boxes: {box_count}")


if __name__ == "__main__":
    main()
