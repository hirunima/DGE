#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


DEFAULT_INPUT_DIR = Path(
    "/fs/nexus-projects/scene_graph_sd/DGE-T2I/data/images/survey_samples"
)
DEFAULT_OUTPUT_DIR_NAME = "survey_samples_resized"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize images in a folder to a shared size."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing images to resize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for resized images. Defaults to a sibling directory "
            "named survey_samples_resized."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Target width. If omitted, uses the first image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Target height. If omitted, uses the first image height.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in the output directory if they already exist.",
    )
    return parser.parse_args()


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def resolve_output_dir(input_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return input_dir.with_name(DEFAULT_OUTPUT_DIR_NAME)


def resolve_target_size(
    images: list[Path], width: int | None, height: int | None
) -> tuple[int, int]:
    if width is not None and height is not None:
        return width, height
    if not images:
        raise ValueError("No images found to infer target size.")
    with Image.open(images[0]) as sample:
        sample_width, sample_height = sample.size
    return (width or sample_width, height or sample_height)


def resize_images(
    images: list[Path],
    input_dir: Path,
    output_dir: Path,
    size: tuple[int, int],
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        relative_path = image_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            continue
        with Image.open(image_path) as image:
            resized = image.resize(size, Image.LANCZOS)
            resized.save(output_path)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = resolve_output_dir(input_dir, args.output_dir)
    images = list_images(input_dir)
    target_size = resolve_target_size(images, args.width, args.height)
    resize_images(images, input_dir, output_dir, target_size, args.overwrite)


if __name__ == "__main__":
    main()
