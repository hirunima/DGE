#!/usr/bin/env python3
"""Lightweight tests for the graph ablation harness."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")

from PIL import Image

from src.eval.ablation import (
    AttributeScorerBackend,
    BackendSpec,
    ExperimentConfig,
    ExperimentItem,
    LabelConfig,
    NodeDetectorBackend,
    RelationScorerBackend,
    STAGE1_VARIANTS,
    STAGE2_VARIANTS,
    STAGE3_VARIANTS,
    StageWeights,
    build_pipeline_permutations,
    compose_score,
    compute_correlation_report,
    config_from_args,
    draw_relation_markers,
    invert_relation,
    normalize_weights,
    parse_stage1_localization,
    prepare_square_crop,
    resolve_siglip_model_path,
    run_ablation_experiment,
    union_bbox,
    write_experiment_outputs,
)


class FakeNodeDetector(NodeDetectorBackend):
    def __init__(self, backend_id, spec, config, score_shift):
        super().__init__(backend_id, spec, config)
        self.score_shift = score_shift

    def detect_nodes(self, image, item):
        nodes = []
        for entity in item.scene_graph["objects"]:
            bbox = [10 + entity["id"] * 5, 12 + entity["id"] * 4, 44 + entity["id"] * 5, 52 + entity["id"] * 4]
            passed = entity["id"] != 2 or self.score_shift > 0
            nodes.append(
                {
                    "id": entity["id"],
                    "name": entity["name"],
                    "bbox": bbox if passed else None,
                    "confidence": 0.9 if passed else 0.1,
                    "passed": passed,
                    "score": 1.0 if passed else 0.0,
                }
            )
        return {
            "backend": self.backend_id,
            "nodes": nodes,
            "fidelity_score": sum(node["score"] for node in nodes) / len(nodes),
        }


class FakeAttributeScorer(AttributeScorerBackend):
    def __init__(self, backend_id, spec, config, score):
        super().__init__(backend_id, spec, config)
        self.score = score

    def score_attributes(self, image, item, stage1_result):
        node_map = {node["id"]: node for node in stage1_result["nodes"]}
        rows = []
        for entity in item.scene_graph["objects"]:
            for attribute in entity.get("attributes", []):
                localized = node_map[entity["id"]]["bbox"] is not None
                rows.append(
                    {
                        "id": entity["id"],
                        "attribute": attribute,
                        "calibrated_score": self.score if localized else None,
                        "score": self.score if localized else None,
                        "skipped": not localized,
                    }
                )
        valid = [row["calibrated_score"] for row in rows if not row["skipped"]]
        return {
            "backend": self.backend_id,
            "attributes": rows,
            "binding_score": sum(valid) / len(valid) if valid else None,
            "skipped_count": sum(1 for row in rows if row["skipped"]),
        }


class FakeRelationScorer(RelationScorerBackend):
    def __init__(self, backend_id, spec, config, score):
        super().__init__(backend_id, spec, config)
        self.score = score

    def score_relations(self, image, item, stage1_result):
        localized = all(node["bbox"] is not None for node in stage1_result["nodes"])
        rows = []
        for relation in item.scene_graph["relations"]:
            if localized:
                rows.append(
                    {
                        "subject": relation["subject"],
                        "relation": relation["relation"],
                        "object": relation["object"],
                        "original_score": self.score,
                        "swapped_score": self.score - 0.3,
                        "delta": 0.3,
                        "swap_correct": True,
                        "skipped": False,
                    }
                )
            else:
                rows.append(
                    {
                        "subject": relation["subject"],
                        "relation": relation["relation"],
                        "object": relation["object"],
                        "original_score": None,
                        "swapped_score": None,
                        "delta": None,
                        "swap_correct": None,
                        "skipped": True,
                    }
                )
        return {
            "backend": self.backend_id,
            "relations": rows,
            "relation_score": self.score if localized else None,
            "swap_accuracy": 1.0 if localized else None,
            "swap_delta_mean": 0.3 if localized else None,
        }


def assert_equal(left, right, message):
    if left != right:
        raise AssertionError(f"{message}: {left!r} != {right!r}")


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def build_test_config(output_dir: str, label_path: str | None = None) -> ExperimentConfig:
    return ExperimentConfig(
        output_dir=output_dir,
        prompts_file=None,
        sg_file=None,
        images_dir=output_dir,
        image_pattern="{index:04d}-{generation}.png",
        generation=1,
        start_idx=0,
        end_idx=None,
        limit=None,
        skip_indices=(),
        weights=StageWeights(0.3, 0.3, 0.3),
        node_confidence_threshold=0.5,
        node_nms_threshold=0.3,
        stage2_crop_size=384,
        stage2_calibration="clip",
        stage2_calibration_scale=1.0,
        stage2_calibration_bias=0.0,
        stage3_margin_ratio=0.1,
        include_model_load_time=False,
        label_config=LabelConfig(path=label_path, key_field="image_id", score_field="score", result_key_field="image_id"),
        backend_specs={stage: BackendSpec("mock") for stage in ("E1", "V1", "E2", "V2", "E3", "V3")},
    )


def build_test_item(image_path: str) -> ExperimentItem:
    return ExperimentItem(
        prompt_index=0,
        image_id="img-1",
        prompt="a red cat above a mat",
        image_path=image_path,
        generation_index=1,
        scene_graph={
            "objects": [
                {"id": 1, "name": "cat", "attributes": ["red"]},
                {"id": 2, "name": "mat", "attributes": ["striped"]},
            ],
            "relations": [{"subject": 1, "relation": "above", "object": 2}],
        },
    )


def test_permutation_matrix_and_weights():
    permutations = build_pipeline_permutations()
    assert_equal(len(permutations), 8, "Expected all 8 permutations")
    weights = normalize_weights(StageWeights(0.3, 0.3, 0.3))
    assert_true(abs(weights["normalized"]["node"] - (1 / 3)) < 1e-9, "Node weight should normalize")
    composed = compose_score({"node": 1.0, "attribute": None, "relation": 0.5}, weights["normalized"])
    assert_true(abs(composed["score"] - 0.75) < 1e-9, "Active weights should renormalize")


def test_bbox_parsing():
    boxes = parse_stage1_localization('{"boxes": [[100, 200, 900, 800]]}', (200, 100))
    assert_equal(boxes[0], [20, 20, 180, 80], "Normalized bbox should convert to pixels")


def test_crop_and_markers():
    image = Image.new("RGB", (100, 60), color=(10, 20, 30))
    crop = prepare_square_crop(image, [10, 10, 50, 30], 384)
    assert_equal(crop.size, (384, 384), "Crop should be resized")
    union = union_bbox([10, 10, 30, 30], [40, 20, 70, 50], 0.1, image.size)
    assert_equal(union, [4, 6, 76, 54], "Union bbox should include margin")
    marked = draw_relation_markers(image, [10, 10, 30, 30], [40, 20, 70, 50])
    assert_equal(marked.size, image.size, "Marked image should preserve size")


def test_relation_swap():
    swapped = invert_relation({"subject": 1, "relation": "chasing", "object": 2})
    assert_equal(swapped, {"subject": 2, "relation": "chasing", "object": 1}, "Relation inversion should swap nodes")


def test_missing_label_file_skip():
    report = compute_correlation_report(
        [{"image_id": "img-1", "final_score": 0.6}],
        None,
        LabelConfig(path="missing.jsonl", key_field="image_id", score_field="score", result_key_field="image_id"),
    )
    assert_equal(report, None, "Missing label data should skip correlation")


def test_siglip_jax_path_resolution():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        jax_dir = base / "siglip2-giant-opt-patch16-384-jax"
        pt_dir = base / "siglip2-giant-opt-patch16-384"
        jax_dir.mkdir()
        pt_dir.mkdir()
        (jax_dir / "weights.npz").write_text("x", encoding="utf-8")
        (pt_dir / "config.json").write_text("{}", encoding="utf-8")
        resolved = resolve_siglip_model_path(str(jax_dir))
        assert_equal(resolved, str(pt_dir), "JAX-only path should resolve to sibling Transformers checkpoint")


def test_reporting_and_smoke_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        image_path = tmp / "image.png"
        Image.new("RGB", (96, 96), color=(120, 120, 120)).save(image_path)
        label_path = tmp / "labels.jsonl"
        label_path.write_text(json.dumps({"image_id": "img-1", "score": 0.8}) + "\n", encoding="utf-8")
        item = build_test_item(str(image_path))
        config = build_test_config(str(tmp), str(label_path))
        backends = {
            "E1": FakeNodeDetector("E1", BackendSpec("fake"), config, 0.0),
            "V1": FakeNodeDetector("V1", BackendSpec("fake"), config, 1.0),
            "E2": FakeAttributeScorer("E2", BackendSpec("fake"), config, 0.4),
            "V2": FakeAttributeScorer("V2", BackendSpec("fake"), config, 0.9),
            "E3": FakeRelationScorer("E3", BackendSpec("fake"), config, 0.3),
            "V3": FakeRelationScorer("V3", BackendSpec("fake"), config, 0.95),
        }
        report = run_ablation_experiment(config, items=[item], backends=backends)
        assert_equal(len(report["aggregate_matrix"]), 8, "Aggregate matrix should cover all permutations")
        matrix = {row["permutation"]: row for row in report["aggregate_matrix"]}
        assert_true(matrix["V1-V2-V3"]["average_final_score"] > matrix["E1-E2-E3"]["average_final_score"], "Better backends should score higher")
        assert_true("V1-V2-V3" in report["correlation_report"], "Correlation report should be present when labels exist")
        paths = write_experiment_outputs(report, str(tmp / "out"))
        assert_true(Path(paths["aggregate_json"]).exists(), "Aggregate JSON should be written")
        assert_true((tmp / "out" / "permutations" / "E1-E2-E3_details.json").exists(), "Detailed results should be written")


def test_cli_config_parsing():
    parser = type("Args", (), {
        "output_dir": "out",
        "prompts_file": "None",
        "sg_file": "graphs.json",
        "images_dir": "images",
        "image_pattern": "{index:04d}-{generation}.png",
        "generation": 1,
        "start_idx": 0,
        "end_idx": 4,
        "limit": 2,
        "skip_indices": "9,20",
        "human_score_file": "labels.json",
        "label_key_field": "sample_id",
        "label_score_field": "human",
        "result_key_field": "prompt_index",
        "weight_node": 0.2,
        "weight_attribute": 0.3,
        "weight_relation": 0.5,
        "node_confidence_threshold": 0.6,
        "node_nms_threshold": 0.2,
        "stage2_crop_size": 256,
        "stage2_calibration": "sigmoid",
        "stage2_calibration_scale": 2.0,
        "stage2_calibration_bias": -0.1,
        "stage3_margin_ratio": 0.2,
        "include_model_load_time": True,
        "e1_backend_kind": "mock",
        "v1_backend_kind": "mock",
        "e2_backend_kind": "mock",
        "v2_backend_kind": "mock",
        "e3_backend_kind": "mock",
        "v3_backend_kind": "mock",
        "eupe_model_path": "eupe",
        "qwen_model_path": "qwen",
        "siglip_model_path": "siglip",
        "llava_model_path": "llava",
        "eupe_checkpoint_path": "eupe.ckpt",
        "qwen_checkpoint_path": "qwen.ckpt",
        "siglip_checkpoint_path": "siglip.ckpt",
        "llava_checkpoint_path": "llava.ckpt",
    })()
    config = config_from_args(parser)
    assert_equal(config.sg_file, "graphs.json", "Scene graph file should be kept")
    assert_equal(config.prompts_file, None, "String None should normalize")
    assert_equal(config.skip_indices, (9, 20), "Skip indices should parse")
    assert_equal(config.weights.relation, 0.5, "Weights should parse")


def main():
    tests = [
        test_permutation_matrix_and_weights,
        test_bbox_parsing,
        test_crop_and_markers,
        test_relation_swap,
        test_missing_label_file_skip,
        test_siglip_jax_path_resolution,
        test_reporting_and_smoke_integration,
        test_cli_config_parsing,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("All graph ablation tests passed.")


if __name__ == "__main__":
    main()
