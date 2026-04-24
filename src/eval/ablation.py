"""Experiment harness for graph-grounded alignment ablations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


STAGE1_VARIANTS = ("E1", "V1")
STAGE2_VARIANTS = ("E2", "V2")
STAGE3_VARIANTS = ("E3", "V3")


def extract_scene_graph(prompt_text: str) -> Dict[str, Any]:
    if "Current Task:" in prompt_text:
        current_task_text = prompt_text.split("Current Task:")[-1]
    else:
        current_task_text = prompt_text
    try:
        obj_section = current_task_text.split("Objects:", 1)[1].split("Relationships:", 1)[0]
        rel_section = current_task_text.split("Relationships:", 1)[1].split("[Step-by-Step Reasoning]", 1)[0]
    except IndexError:
        return {"error": "Could not find Objects or Relationships sections in the expected format."}

    objects = []
    current_object = None
    for line in obj_section.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("-") and "(object id" in stripped:
            prefix, suffix = stripped[1:].split("(object id", 1)
            name = prefix.strip().split(" ", 1)[-1].strip()
            identifier = int(suffix.split(":")[-1].split(")")[0].strip())
            current_object = {"id": identifier, "name": name, "attributes": []}
            objects.append(current_object)
            continue
        if current_object is not None and stripped.startswith("-"):
            current_object["attributes"].append(stripped[1:].strip())

    relations = []
    for line in rel_section.strip().splitlines():
        stripped = line.strip()
        if not stripped.startswith("- Object"):
            continue
        tokens = stripped[1:].strip().split()
        if len(tokens) < 5:
            continue
        subject = int(tokens[1])
        object_id = int(tokens[-1])
        relation = " ".join(tokens[2:-2])
        relations.append({"subject": subject, "relation": relation, "object": object_id})

    return {"objects": objects, "relations": relations}


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def resolve_siglip_model_path(model_path: str) -> str:
    path = Path(model_path)
    if path.is_dir() and (path / "config.json").exists():
        return str(path)
    if path.is_dir() and any(path.glob("*.npz")):
        sibling = path.parent / path.name.replace("-jax", "")
        if sibling.exists() and (sibling / "config.json").exists():
            return str(sibling)
        raise ValueError(
            f"SigLIP path {model_path} only contains JAX weights. "
            f"Download a Transformers-compatible checkpoint such as {sibling}."
        )
    return model_path


@dataclass(frozen=True)
class StageWeights:
    node: float
    attribute: float
    relation: float


@dataclass(frozen=True)
class ExperimentItem:
    prompt_index: int
    image_id: str
    prompt: str
    image_path: str
    scene_graph: Dict[str, Any]
    generation_index: Optional[int] = None


@dataclass(frozen=True)
class LabelConfig:
    path: Optional[str]
    key_field: str = "image_id"
    score_field: str = "score"
    result_key_field: str = "image_id"


@dataclass(frozen=True)
class BackendSpec:
    kind: str
    model_path: Optional[str] = None
    checkpoint_path: Optional[str] = None


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: str
    prompts_file: Optional[str]
    sg_file: Optional[str]
    images_dir: str
    image_pattern: str
    generation: int
    start_idx: int
    end_idx: Optional[int]
    limit: Optional[int]
    weights: StageWeights
    node_confidence_threshold: float
    node_nms_threshold: float
    stage2_crop_size: int
    stage2_calibration: str
    stage2_calibration_scale: float
    stage2_calibration_bias: float
    stage3_margin_ratio: float
    include_model_load_time: bool
    label_config: LabelConfig
    backend_specs: Dict[str, BackendSpec]


class NodeDetectorBackend(ABC):
    def __init__(self, backend_id: str, spec: BackendSpec, config: ExperimentConfig) -> None:
        self.backend_id = backend_id
        self.spec = spec
        self.config = config
        self.model_load_time_ms = 0.0

    @abstractmethod
    def detect_nodes(self, image: Image.Image, item: ExperimentItem) -> Dict[str, Any]:
        raise NotImplementedError


class AttributeScorerBackend(ABC):
    def __init__(self, backend_id: str, spec: BackendSpec, config: ExperimentConfig) -> None:
        self.backend_id = backend_id
        self.spec = spec
        self.config = config
        self.model_load_time_ms = 0.0

    @abstractmethod
    def score_attributes(
        self,
        image: Image.Image,
        item: ExperimentItem,
        stage1_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError


class RelationScorerBackend(ABC):
    def __init__(self, backend_id: str, spec: BackendSpec, config: ExperimentConfig) -> None:
        self.backend_id = backend_id
        self.spec = spec
        self.config = config
        self.model_load_time_ms = 0.0

    @abstractmethod
    def score_relations(
        self,
        image: Image.Image,
        item: ExperimentItem,
        stage1_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError


class UnavailableNodeDetector(NodeDetectorBackend):
    def detect_nodes(self, image: Image.Image, item: ExperimentItem) -> Dict[str, Any]:
        raise NotImplementedError(
            f"Backend {self.backend_id} is configured as '{self.spec.kind}' but acquisition is not implemented yet."
        )


class UnavailableAttributeScorer(AttributeScorerBackend):
    def score_attributes(
        self,
        image: Image.Image,
        item: ExperimentItem,
        stage1_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            f"Backend {self.backend_id} is configured as '{self.spec.kind}' but acquisition is not implemented yet."
        )


class UnavailableRelationScorer(RelationScorerBackend):
    def score_relations(
        self,
        image: Image.Image,
        item: ExperimentItem,
        stage1_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            f"Backend {self.backend_id} is configured as '{self.spec.kind}' but acquisition is not implemented yet."
        )


class _TransformersBackendMixin:
    def _import_torch(self) -> Any:
        import torch

        return torch

    def _load_transformers_components(self) -> Tuple[Any, Any]:
        from transformers import AutoModel, AutoModelForImageTextToText, AutoModelForZeroShotObjectDetection, AutoProcessor

        return AutoProcessor, AutoModel, AutoModelForImageTextToText, AutoModelForZeroShotObjectDetection

    def _ensure_model_path(self) -> str:
        if not self.spec.model_path:
            raise ValueError(f"Backend {self.backend_id} requires --model-path input for kind '{self.spec.kind}'.")
        return self.spec.model_path

    def _move_batch_to_device(self, batch: Mapping[str, Any], device: Any) -> Dict[str, Any]:
        moved = {}
        for key, value in batch.items():
            if hasattr(value, "to"):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    def _extract_device(self, model: Any) -> Any:
        device = getattr(model, "device", None)
        if device is not None:
            return device
        return next(model.parameters()).device

    def _load_processor_and_model(
        self,
        model_loader: Any,
        extra_model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Any]:
        torch = self._import_torch()
        AutoProcessor, _, _, _ = self._load_transformers_components()
        model_path = self._ensure_model_path()
        load_start = time.perf_counter()
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": "auto",
            "device_map": "auto" if torch.cuda.is_available() else None,
            "low_cpu_mem_usage": True,
        }
        if extra_model_kwargs:
            model_kwargs.update(extra_model_kwargs)
        if model_kwargs.get("device_map") is None:
            model_kwargs.pop("device_map")
        model = model_loader.from_pretrained(model_path, **model_kwargs)
        self._maybe_load_checkpoint(model)
        model.eval()
        self.model_load_time_ms = (time.perf_counter() - load_start) * 1000.0
        return processor, model

    def _maybe_load_checkpoint(self, model: Any) -> None:
        if not self.spec.checkpoint_path:
            return
        torch = self._import_torch()
        checkpoint = torch.load(self.spec.checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint {self.spec.checkpoint_path} does not contain a state dict.")
        cleaned = {}
        for key, value in checkpoint.items():
            cleaned[key.replace("module.", "", 1)] = value
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        model._ablation_checkpoint_info = {
            "checkpoint_path": self.spec.checkpoint_path,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }


class GroundingNodeDetector(NodeDetectorBackend, _TransformersBackendMixin):
    def __init__(self, backend_id: str, spec: BackendSpec, config: ExperimentConfig) -> None:
        super().__init__(backend_id, spec, config)
        _, _, _, AutoModelForZeroShotObjectDetection = self._load_transformers_components()
        self.processor, self.model = self._load_processor_and_model(AutoModelForZeroShotObjectDetection)

    def _normalize_label(self, label: Any) -> str:
        if isinstance(label, str):
            return label.strip().lower()
        if isinstance(label, int):
            id2label = getattr(getattr(self.model, "config", None), "id2label", {})
            return str(id2label.get(label, label)).strip().lower()
        return str(label).strip().lower()

    def _run_detector(self, image: Image.Image, text: str) -> List[Dict[str, Any]]:
        torch = self._import_torch()
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = self._move_batch_to_device(inputs, self._extract_device(self.model))
        with torch.inference_mode():
            outputs = self.model(**inputs)
        if not hasattr(self.processor, "post_process_grounded_object_detection"):
            raise NotImplementedError(
                f"Processor for {self.spec.model_path} does not expose post_process_grounded_object_detection."
            )
        target_sizes = torch.tensor([[image.size[1], image.size[0]]], device=self._extract_device(self.model))
        processed = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.get("input_ids"),
            threshold=self.config.node_confidence_threshold,
            target_sizes=target_sizes,
        )[0]
        boxes = processed.get("boxes", [])
        scores = processed.get("scores", [])
        labels = processed.get("labels", [])
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            bbox = clamp_bbox(box.detach().float().cpu().tolist(), image.size[0], image.size[1])
            detections.append(
                {
                    "bbox": bbox,
                    "confidence": float(score.detach().float().cpu().item()),
                    "label": self._normalize_label(label),
                }
            )
        return detections

    def detect_nodes(self, image: Image.Image, item: ExperimentItem) -> Dict[str, Any]:
        entity_names = [entity.get("name", "") for entity in item.scene_graph.get("objects", [])]
        detection_prompt = " . ".join(name.strip() for name in entity_names if name and name.strip())
        detections = self._run_detector(image, detection_prompt)
        nodes = []
        for entity in item.scene_graph.get("objects", []):
            entity_name = str(entity.get("name", "")).strip().lower()
            candidates = [
                det
                for det in detections
                if entity_name == det["label"] or entity_name in det["label"] or det["label"] in entity_name
            ]
            best = max(candidates, key=lambda row: row["confidence"], default=None)
            passed = bool(best and best["confidence"] >= self.config.node_confidence_threshold)
            nodes.append(
                {
                    "id": entity.get("id"),
                    "name": entity.get("name"),
                    "bbox": best["bbox"] if passed else None,
                    "confidence": best["confidence"] if best else 0.0,
                    "passed": passed,
                    "score": 1.0 if passed else 0.0,
                    "raw_output": best,
                }
            )
        return {
            "backend": self.backend_id,
            "thresholds": {
                "confidence": self.config.node_confidence_threshold,
                "nms": self.config.node_nms_threshold,
            },
            "detections": detections,
            "nodes": nodes,
            "fidelity_score": safe_mean(entry["score"] for entry in nodes),
        }


class EupeNodeDetector(NodeDetectorBackend):
    def __init__(self, backend_id: str, spec: BackendSpec, config: ExperimentConfig) -> None:
        super().__init__(backend_id, spec, config)
        self.repo_dir = self._resolve_repo_dir()
        self.checkpoint_path = self._resolve_checkpoint_path()
        self.model = self._load_model()

    def _resolve_repo_dir(self) -> str:
        if self.spec.model_path and Path(self.spec.model_path).is_dir() and (Path(self.spec.model_path) / "hubconf.py").exists():
            return self.spec.model_path
        default_repo = Path("/fs/nexus-projects/scene_graph_sd/eval_checkpoints/facebook/eupe_repo")
        if default_repo.exists():
            return str(default_repo)
        raise ValueError(
            f"EUPE backend requires the cloned facebookresearch/eupe repo. "
            f"Pass --eupe-model-path pointing at the repo checkout."
        )

    def _resolve_checkpoint_path(self) -> str:
        if self.spec.checkpoint_path and Path(self.spec.checkpoint_path).exists():
            return self.spec.checkpoint_path
        if self.spec.model_path:
            candidate = Path(self.spec.model_path)
            if candidate.is_file():
                return str(candidate)
            if candidate.is_dir():
                pt_files = sorted(candidate.glob("*.pt"))
                if pt_files:
                    return str(pt_files[0])
        default_ckpt = Path("/fs/nexus-projects/scene_graph_sd/eval_checkpoints/facebook/EUPE-ViT-B/EUPE-ViT-B.pt")
        if default_ckpt.exists():
            return str(default_ckpt)
        raise ValueError(
            f"EUPE backend requires a .pt checkpoint. Pass --eupe-checkpoint-path or a directory containing the checkpoint."
        )

    def _load_model(self) -> Any:
        import sys
        import torch

        load_start = time.perf_counter()
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)
        from eupe.hub.backbones import eupe_vitb16

        model = eupe_vitb16(pretrained=False)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise ValueError(f"EUPE checkpoint {self.checkpoint_path} does not contain a state dict.")
        cleaned = {key.replace("module.", "", 1): value for key, value in checkpoint.items()}
        model.load_state_dict(cleaned, strict=False)
        model.eval()
        self.model_load_time_ms = (time.perf_counter() - load_start) * 1000.0
        return model

    def detect_nodes(self, image: Image.Image, item: ExperimentItem) -> Dict[str, Any]:
        raise NotImplementedError(
            "facebook/EUPE-ViT-B loads successfully as a vision backbone, but it is not a text-grounded detector. "
            "This exact checkpoint cannot perform Stage 1 node localization by itself. "
            "Use a grounding-capable checkpoint for E1 or change the experiment design."
        )


class _VisionLanguageBackend(_TransformersBackendMixin):
    def __init__(self, backend_id: str, spec: BackendSpec, config: ExperimentConfig) -> None:
        self.backend_id = backend_id
        self.spec = spec
        self.config = config
        self.model_load_time_ms = 0.0
        _, _, AutoModelForImageTextToText, _ = self._load_transformers_components()
        self.processor, self.model = self._load_processor_and_model(AutoModelForImageTextToText)

    def _build_messages(self, image: Image.Image, prompt: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _prepare_chat_inputs(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        messages = self._build_messages(image, prompt)
        if not hasattr(self.processor, "apply_chat_template"):
            raise NotImplementedError(f"Processor for {self.spec.model_path} does not expose apply_chat_template.")
        batch = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return self._move_batch_to_device(batch, self._extract_device(self.model))

    def generate_text(self, image: Image.Image, prompt: str, max_new_tokens: int = 128) -> str:
        torch = self._import_torch()
        batch = self._prepare_chat_inputs(image, prompt)
        with torch.inference_mode():
            outputs = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = outputs[:, batch["input_ids"].shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def yes_no_score(self, image: Image.Image, prompt: str) -> Tuple[float, Dict[str, Any]]:
        torch = self._import_torch()
        batch = self._prepare_chat_inputs(image, prompt)
        with torch.inference_mode():
            outputs = self.model.generate(
                **batch,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        logits = outputs.scores[0][0]
        yes_token = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token = self.processor.tokenizer.encode("No", add_special_tokens=False)[0]
        selected = torch.stack([logits[yes_token], logits[no_token]])
        probs = torch.softmax(selected, dim=0).detach().float().cpu().tolist()
        return float(probs[0]), {"yes_prob": float(probs[0]), "no_prob": float(probs[1])}


class QwenNodeDetector(NodeDetectorBackend, _VisionLanguageBackend):
    def __init__(self, backend_id: str, spec: BackendSpec, config: ExperimentConfig) -> None:
        NodeDetectorBackend.__init__(self, backend_id, spec, config)
        _VisionLanguageBackend.__init__(self, backend_id, spec, config)

    def _localization_prompt(self, entity: Mapping[str, Any], image: Image.Image) -> str:
        return (
            "Return JSON only.\n"
            "Locate the requested object in the image.\n"
            "Use normalized coordinates in [0,1000].\n"
            "If the object is absent, return {\"boxes\": [], \"confidence\": 0.0}.\n"
            "Otherwise return {\"boxes\": [[x1, y1, x2, y2]], \"confidence\": 1.0}.\n"
            f"Image size: width={image.size[0]}, height={image.size[1]}.\n"
            f"Object: {entity.get('name')}\n"
            f"Attributes: {json.dumps(entity.get('attributes') or [])}\n"
        )

    def detect_nodes(self, image: Image.Image, item: ExperimentItem) -> Dict[str, Any]:
        nodes = []
        for entity in item.scene_graph.get("objects", []):
            raw_text = self.generate_text(image, self._localization_prompt(entity, image), max_new_tokens=128)
            try:
                parsed = json.loads(raw_text[raw_text.find("{") : raw_text.rfind("}") + 1]) if "{" in raw_text else {}
            except json.JSONDecodeError:
                parsed = {}
            boxes = parse_stage1_localization(json.dumps(parsed if parsed else {"boxes": []}), image.size)
            confidence = float(parsed.get("confidence", 1.0 if boxes else 0.0))
            passed = bool(boxes) and confidence >= self.config.node_confidence_threshold
            nodes.append(
                {
                    "id": entity.get("id"),
                    "name": entity.get("name"),
                    "bbox": boxes[0] if passed else None,
                    "confidence": confidence,
                    "passed": passed,
                    "score": 1.0 if passed else 0.0,
                    "raw_output": raw_text,
                }
            )
        return {
            "backend": self.backend_id,
            "thresholds": {
                "confidence": self.config.node_confidence_threshold,
                "nms": self.config.node_nms_threshold,
            },
            "nodes": nodes,
            "fidelity_score": safe_mean(entry["score"] for entry in nodes),
        }


class SigLIPAttributeScorer(AttributeScorerBackend, _TransformersBackendMixin):
    def __init__(self, backend_id: str, spec: BackendSpec, config: ExperimentConfig) -> None:
        super().__init__(backend_id, spec, config)
        if self.spec.model_path:
            self.spec = BackendSpec(self.spec.kind, resolve_siglip_model_path(self.spec.model_path), self.spec.checkpoint_path)
        _, AutoModel, _, _ = self._load_transformers_components()
        self.processor, self.model = self._load_processor_and_model(AutoModel)

    def _image_text_similarity(self, image: Image.Image, text: str) -> float:
        torch = self._import_torch()
        batch = self.processor(images=image, text=[text], return_tensors="pt", padding=True)
        batch = self._move_batch_to_device(batch, self._extract_device(self.model))
        with torch.inference_mode():
            if hasattr(self.model, "get_image_features") and hasattr(self.model, "get_text_features"):
                image_features = self.model.get_image_features(pixel_values=batch["pixel_values"])
                text_features = self.model.get_text_features(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                )
            else:
                outputs = self.model(**batch)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        similarity = (image_features @ text_features.transpose(0, 1))[0, 0]
        return float(similarity.detach().float().cpu().item())

    def score_attributes(
        self,
        image: Image.Image,
        item: ExperimentItem,
        stage1_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        node_map = {node["id"]: node for node in stage1_result.get("nodes", [])}
        results = []
        for entity in item.scene_graph.get("objects", []):
            for attribute in entity.get("attributes") or []:
                node_result = node_map.get(entity.get("id"))
                if not node_result or not node_result.get("passed") or not node_result.get("bbox"):
                    results.append(
                        {
                            "id": entity.get("id"),
                            "name": entity.get("name"),
                            "attribute": attribute,
                            "score": None,
                            "calibrated_score": None,
                            "skipped": True,
                            "skip_reason": "node_not_localized",
                        }
                    )
                    continue
                crop = prepare_square_crop(image, node_result["bbox"], self.config.stage2_crop_size)
                text = f"A photo of a {attribute} {entity.get('name')}"
                raw_score = self._image_text_similarity(crop, text)
                score = calibrate_score(
                    raw_score,
                    self.config.stage2_calibration,
                    self.config.stage2_calibration_scale,
                    self.config.stage2_calibration_bias,
                )
                results.append(
                    {
                        "id": entity.get("id"),
                        "name": entity.get("name"),
                        "attribute": attribute,
                        "score": raw_score,
                        "calibrated_score": score,
                        "skipped": False,
                        "skip_reason": None,
                        "bbox": node_result["bbox"],
                    }
                )
        return {
            "backend": self.backend_id,
            "crop_size": self.config.stage2_crop_size,
            "calibration": {
                "name": self.config.stage2_calibration,
                "scale": self.config.stage2_calibration_scale,
                "bias": self.config.stage2_calibration_bias,
            },
            "attributes": results,
            "binding_score": safe_mean(entry["calibrated_score"] for entry in results if not entry.get("skipped")),
            "skipped_count": sum(1 for entry in results if entry.get("skipped")),
        }


class LlavaAttributeScorer(AttributeScorerBackend, _VisionLanguageBackend):
    def __init__(self, backend_id: str, spec: BackendSpec, config: ExperimentConfig) -> None:
        AttributeScorerBackend.__init__(self, backend_id, spec, config)
        _VisionLanguageBackend.__init__(self, backend_id, spec, config)

    def score_attributes(
        self,
        image: Image.Image,
        item: ExperimentItem,
        stage1_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        node_map = {node["id"]: node for node in stage1_result.get("nodes", [])}
        results = []
        for entity in item.scene_graph.get("objects", []):
            for attribute in entity.get("attributes") or []:
                node_result = node_map.get(entity.get("id"))
                if not node_result or not node_result.get("passed") or not node_result.get("bbox"):
                    results.append(
                        {
                            "id": entity.get("id"),
                            "name": entity.get("name"),
                            "attribute": attribute,
                            "score": None,
                            "calibrated_score": None,
                            "skipped": True,
                            "skip_reason": "node_not_localized",
                        }
                    )
                    continue
                crop = prepare_square_crop(image, node_result["bbox"], self.config.stage2_crop_size)
                prompt = (
                    "Answer strictly with Yes or No.\n"
                    f"Does the highlighted crop contain a {entity.get('name')} with attribute '{attribute}'?"
                )
                yes_score, probs = self.yes_no_score(crop, prompt)
                results.append(
                    {
                        "id": entity.get("id"),
                        "name": entity.get("name"),
                        "attribute": attribute,
                        "score": yes_score,
                        "calibrated_score": yes_score,
                        "token_probs": probs,
                        "skipped": False,
                        "skip_reason": None,
                        "bbox": node_result["bbox"],
                    }
                )
        return {
            "backend": self.backend_id,
            "crop_size": self.config.stage2_crop_size,
            "attributes": results,
            "binding_score": safe_mean(entry["calibrated_score"] for entry in results if not entry.get("skipped")),
            "skipped_count": sum(1 for entry in results if entry.get("skipped")),
        }


class QwenAttributeScorer(AttributeScorerBackend, _VisionLanguageBackend):
    def __init__(self, backend_id: str, spec: BackendSpec, config: ExperimentConfig) -> None:
        AttributeScorerBackend.__init__(self, backend_id, spec, config)
        _VisionLanguageBackend.__init__(self, backend_id, spec, config)

    def score_attributes(
        self,
        image: Image.Image,
        item: ExperimentItem,
        stage1_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        node_map = {node["id"]: node for node in stage1_result.get("nodes", [])}
        results = []
        for entity in item.scene_graph.get("objects", []):
            for attribute in entity.get("attributes") or []:
                node_result = node_map.get(entity.get("id"))
                if not node_result or not node_result.get("passed") or not node_result.get("bbox"):
                    results.append(
                        {
                            "id": entity.get("id"),
                            "name": entity.get("name"),
                            "attribute": attribute,
                            "score": None,
                            "calibrated_score": None,
                            "skipped": True,
                            "skip_reason": "node_not_localized",
                        }
                    )
                    continue
                crop = prepare_square_crop(image, node_result["bbox"], self.config.stage2_crop_size)
                prompt = (
                    "Answer strictly with Yes or No.\n"
                    f"Does this crop show a {entity.get('name')} with attribute '{attribute}'?"
                )
                yes_score, probs = self.yes_no_score(crop, prompt)
                results.append(
                    {
                        "id": entity.get("id"),
                        "name": entity.get("name"),
                        "attribute": attribute,
                        "score": yes_score,
                        "calibrated_score": yes_score,
                        "token_probs": probs,
                        "skipped": False,
                        "skip_reason": None,
                        "bbox": node_result["bbox"],
                    }
                )
        return {
            "backend": self.backend_id,
            "crop_size": self.config.stage2_crop_size,
            "attributes": results,
            "binding_score": safe_mean(entry["calibrated_score"] for entry in results if not entry.get("skipped")),
            "skipped_count": sum(1 for entry in results if entry.get("skipped")),
        }


class SigLIPRelationScorer(RelationScorerBackend, _TransformersBackendMixin):
    def __init__(self, backend_id: str, spec: BackendSpec, config: ExperimentConfig) -> None:
        super().__init__(backend_id, spec, config)
        if self.spec.model_path:
            self.spec = BackendSpec(self.spec.kind, resolve_siglip_model_path(self.spec.model_path), self.spec.checkpoint_path)
        _, AutoModel, _, _ = self._load_transformers_components()
        self.processor, self.model = self._load_processor_and_model(AutoModel)

    def _image_text_similarity(self, image: Image.Image, text: str) -> float:
        torch = self._import_torch()
        batch = self.processor(images=image, text=[text], return_tensors="pt", padding=True)
        batch = self._move_batch_to_device(batch, self._extract_device(self.model))
        with torch.inference_mode():
            if hasattr(self.model, "get_image_features") and hasattr(self.model, "get_text_features"):
                image_features = self.model.get_image_features(pixel_values=batch["pixel_values"])
                text_features = self.model.get_text_features(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                )
            else:
                outputs = self.model(**batch)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        similarity = (image_features @ text_features.transpose(0, 1))[0, 0]
        return float(similarity.detach().float().cpu().item())

    def score_relations(
        self,
        image: Image.Image,
        item: ExperimentItem,
        stage1_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        node_map = {node["id"]: node for node in stage1_result.get("nodes", [])}
        entity_map = {entity["id"]: entity for entity in item.scene_graph.get("objects", [])}
        results = []
        for relation in item.scene_graph.get("relations", []):
            subj = node_map.get(relation.get("subject"))
            obj = node_map.get(relation.get("object"))
            if not subj or not obj or not subj.get("bbox") or not obj.get("bbox"):
                results.append(
                    {
                        "subject": relation.get("subject"),
                        "relation": relation.get("relation"),
                        "object": relation.get("object"),
                        "original_score": None,
                        "swapped_score": None,
                        "delta": None,
                        "swap_correct": None,
                        "skipped": True,
                        "skip_reason": "missing_localization",
                    }
                )
                continue
            crop_box = union_bbox(subj["bbox"], obj["bbox"], self.config.stage3_margin_ratio, image.size)
            crop = image.crop(tuple(crop_box))
            subj_name = entity_map.get(relation.get("subject"), {}).get("name", "subject")
            obj_name = entity_map.get(relation.get("object"), {}).get("name", "object")
            original_text = f"{subj_name} {relation.get('relation')} {obj_name}"
            swapped_text = f"{obj_name} {relation.get('relation')} {subj_name}"
            original_score = calibrate_score(
                self._image_text_similarity(crop, original_text),
                self.config.stage2_calibration,
                self.config.stage2_calibration_scale,
                self.config.stage2_calibration_bias,
            )
            swapped_score = calibrate_score(
                self._image_text_similarity(crop, swapped_text),
                self.config.stage2_calibration,
                self.config.stage2_calibration_scale,
                self.config.stage2_calibration_bias,
            )
            results.append(
                {
                    "subject": relation.get("subject"),
                    "relation": relation.get("relation"),
                    "object": relation.get("object"),
                    "swapped_subject": relation.get("object"),
                    "swapped_object": relation.get("subject"),
                    "original_score": original_score,
                    "swapped_score": swapped_score,
                    "delta": original_score - swapped_score,
                    "swap_correct": original_score > swapped_score,
                    "skipped": False,
                    "skip_reason": None,
                    "union_bbox": crop_box,
                }
            )
        return {
            "backend": self.backend_id,
            "margin_ratio": self.config.stage3_margin_ratio,
            "relations": results,
            "relation_score": safe_mean(entry["original_score"] for entry in results if not entry.get("skipped")),
            "swap_accuracy": safe_mean(
                1.0 if entry.get("swap_correct") else 0.0 for entry in results if entry.get("swap_correct") is not None
            ),
            "swap_delta_mean": safe_mean(entry["delta"] for entry in results if entry.get("delta") is not None),
        }


class QwenRelationScorer(RelationScorerBackend, _VisionLanguageBackend):
    def __init__(self, backend_id: str, spec: BackendSpec, config: ExperimentConfig) -> None:
        RelationScorerBackend.__init__(self, backend_id, spec, config)
        _VisionLanguageBackend.__init__(self, backend_id, spec, config)

    def score_relations(
        self,
        image: Image.Image,
        item: ExperimentItem,
        stage1_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        node_map = {node["id"]: node for node in stage1_result.get("nodes", [])}
        entity_map = {entity["id"]: entity for entity in item.scene_graph.get("objects", [])}
        results = []
        for relation in item.scene_graph.get("relations", []):
            subj = node_map.get(relation.get("subject"))
            obj = node_map.get(relation.get("object"))
            if not subj or not obj or not subj.get("bbox") or not obj.get("bbox"):
                results.append(
                    {
                        "subject": relation.get("subject"),
                        "relation": relation.get("relation"),
                        "object": relation.get("object"),
                        "original_score": None,
                        "swapped_score": None,
                        "delta": None,
                        "swap_correct": None,
                        "skipped": True,
                        "skip_reason": "missing_localization",
                    }
                )
                continue
            marked = draw_relation_markers(image, subj["bbox"], obj["bbox"])
            subj_name = entity_map.get(relation.get("subject"), {}).get("name", "subject")
            obj_name = entity_map.get(relation.get("object"), {}).get("name", "object")
            original_prompt = (
                "Answer strictly with Yes or No.\n"
                "The red box marks the subject and the blue box marks the object.\n"
                f"Is the relation true: {subj_name} {relation.get('relation')} {obj_name}?"
            )
            swapped_prompt = (
                "Answer strictly with Yes or No.\n"
                "The red box marks the subject and the blue box marks the object.\n"
                f"Is the relation true: {obj_name} {relation.get('relation')} {subj_name}?"
            )
            original_score, original_probs = self.yes_no_score(marked, original_prompt)
            swapped_score, swapped_probs = self.yes_no_score(marked, swapped_prompt)
            results.append(
                {
                    "subject": relation.get("subject"),
                    "relation": relation.get("relation"),
                    "object": relation.get("object"),
                    "swapped_subject": relation.get("object"),
                    "swapped_object": relation.get("subject"),
                    "original_score": original_score,
                    "swapped_score": swapped_score,
                    "delta": original_score - swapped_score,
                    "swap_correct": original_score > swapped_score,
                    "skipped": False,
                    "skip_reason": None,
                    "token_probs": {
                        "original": original_probs,
                        "swapped": swapped_probs,
                    },
                    "marker_mode": marked.mode,
                }
            )
        return {
            "backend": self.backend_id,
            "margin_ratio": self.config.stage3_margin_ratio,
            "relations": results,
            "relation_score": safe_mean(entry["original_score"] for entry in results if not entry.get("skipped")),
            "swap_accuracy": safe_mean(
                1.0 if entry.get("swap_correct") else 0.0 for entry in results if entry.get("swap_correct") is not None
            ),
            "swap_delta_mean": safe_mean(entry["delta"] for entry in results if entry.get("delta") is not None),
        }


def _stable_unit(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


class MockNodeDetector(NodeDetectorBackend):
    def detect_nodes(self, image: Image.Image, item: ExperimentItem) -> Dict[str, Any]:
        width, height = image.size
        threshold = self.config.node_confidence_threshold
        nodes = []
        for entity in item.scene_graph.get("objects", []):
            confidence = 0.35 + 0.64 * _stable_unit(item.image_id, self.backend_id, entity.get("id"), entity.get("name"))
            if self.backend_id == "V1":
                normalized_box = [
                    round(50 + 650 * _stable_unit(entity.get("id"), "x1"), 2),
                    round(60 + 600 * _stable_unit(entity.get("id"), "y1"), 2),
                    round(700 + 200 * _stable_unit(entity.get("id"), "x2"), 2),
                    round(680 + 180 * _stable_unit(entity.get("id"), "y2"), 2),
                ]
                bbox = normalized_bbox_to_pixel(normalized_box, width, height)
                raw = json.dumps({"boxes": [normalized_box], "confidence": confidence})
            else:
                x1 = int(width * (0.05 + 0.4 * _stable_unit(entity.get("id"), "px1")))
                y1 = int(height * (0.05 + 0.4 * _stable_unit(entity.get("id"), "py1")))
                box_width = max(10, int(width * (0.18 + 0.18 * _stable_unit(entity.get("id"), "pw"))))
                box_height = max(10, int(height * (0.18 + 0.18 * _stable_unit(entity.get("id"), "ph"))))
                bbox = clamp_bbox([x1, y1, x1 + box_width, y1 + box_height], width, height)
                raw = {"bbox": bbox, "confidence": confidence}
            passed = confidence >= threshold
            nodes.append(
                {
                    "id": entity.get("id"),
                    "name": entity.get("name"),
                    "bbox": bbox if passed else None,
                    "confidence": confidence,
                    "passed": passed,
                    "score": 1.0 if passed else 0.0,
                    "raw_output": raw,
                }
            )

        return {
            "backend": self.backend_id,
            "thresholds": {
                "confidence": self.config.node_confidence_threshold,
                "nms": self.config.node_nms_threshold,
            },
            "nodes": nodes,
            "fidelity_score": safe_mean(entry["score"] for entry in nodes),
        }


class MockAttributeScorer(AttributeScorerBackend):
    def score_attributes(
        self,
        image: Image.Image,
        item: ExperimentItem,
        stage1_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        node_map = {node["id"]: node for node in stage1_result.get("nodes", [])}
        results = []
        for entity in item.scene_graph.get("objects", []):
            for attribute in entity.get("attributes") or []:
                node_result = node_map.get(entity.get("id"))
                if not node_result or not node_result.get("passed") or not node_result.get("bbox"):
                    results.append(
                        {
                            "id": entity.get("id"),
                            "name": entity.get("name"),
                            "attribute": attribute,
                            "score": None,
                            "calibrated_score": None,
                            "skipped": True,
                            "skip_reason": "node_not_localized",
                        }
                    )
                    continue
                crop = prepare_square_crop(image, node_result["bbox"], self.config.stage2_crop_size)
                score = calibrate_score(
                    0.1 + 0.8 * _stable_unit(item.image_id, self.backend_id, entity.get("id"), attribute, crop.size),
                    self.config.stage2_calibration,
                    self.config.stage2_calibration_scale,
                    self.config.stage2_calibration_bias,
                )
                results.append(
                    {
                        "id": entity.get("id"),
                        "name": entity.get("name"),
                        "attribute": attribute,
                        "score": score,
                        "calibrated_score": score,
                        "skipped": False,
                        "skip_reason": None,
                        "bbox": node_result["bbox"],
                    }
                )

        return {
            "backend": self.backend_id,
            "crop_size": self.config.stage2_crop_size,
            "calibration": {
                "name": self.config.stage2_calibration,
                "scale": self.config.stage2_calibration_scale,
                "bias": self.config.stage2_calibration_bias,
            },
            "attributes": results,
            "binding_score": safe_mean(
                entry["calibrated_score"] for entry in results if not entry.get("skipped")
            ),
            "skipped_count": sum(1 for entry in results if entry.get("skipped")),
        }


class MockRelationScorer(RelationScorerBackend):
    def score_relations(
        self,
        image: Image.Image,
        item: ExperimentItem,
        stage1_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        node_map = {node["id"]: node for node in stage1_result.get("nodes", [])}
        results = []
        for relation in item.scene_graph.get("relations", []):
            subj = node_map.get(relation.get("subject"))
            obj = node_map.get(relation.get("object"))
            if not subj or not obj or not subj.get("bbox") or not obj.get("bbox"):
                results.append(
                    {
                        "subject": relation.get("subject"),
                        "relation": relation.get("relation"),
                        "object": relation.get("object"),
                        "original_score": None,
                        "swapped_score": None,
                        "delta": None,
                        "swap_correct": None,
                        "skipped": True,
                        "skip_reason": "missing_localization",
                    }
                )
                continue

            union_box = union_bbox(subj["bbox"], obj["bbox"], self.config.stage3_margin_ratio, image.size)
            original_score = 0.1 + 0.8 * _stable_unit(
                item.image_id,
                self.backend_id,
                relation.get("subject"),
                relation.get("relation"),
                relation.get("object"),
                tuple(union_box),
            )
            swapped_relation = invert_relation(relation)
            swapped_score = max(
                0.0,
                min(
                    1.0,
                    original_score
                    - (0.1 + 0.25 * _stable_unit(item.image_id, self.backend_id, "delta", relation.get("relation"))),
                ),
            )

            if self.backend_id == "V3":
                marked = draw_relation_markers(image, subj["bbox"], obj["bbox"])
                marker_mode = marked.mode
            else:
                marker_mode = None

            results.append(
                {
                    "subject": relation.get("subject"),
                    "relation": relation.get("relation"),
                    "object": relation.get("object"),
                    "swapped_subject": swapped_relation.get("subject"),
                    "swapped_object": swapped_relation.get("object"),
                    "original_score": original_score,
                    "swapped_score": swapped_score,
                    "delta": original_score - swapped_score,
                    "swap_correct": original_score > swapped_score,
                    "skipped": False,
                    "skip_reason": None,
                    "union_bbox": union_box,
                    "marker_mode": marker_mode,
                }
            )

        return {
            "backend": self.backend_id,
            "margin_ratio": self.config.stage3_margin_ratio,
            "relations": results,
            "relation_score": safe_mean(
                entry["original_score"] for entry in results if not entry.get("skipped")
            ),
            "swap_accuracy": safe_mean(
                1.0 if entry.get("swap_correct") else 0.0
                for entry in results
                if entry.get("swap_correct") is not None
            ),
            "swap_delta_mean": safe_mean(
                entry["delta"] for entry in results if entry.get("delta") is not None
            ),
        }


def build_backend(backend_id: str, spec: BackendSpec, config: ExperimentConfig) -> Any:
    kind = spec.kind.lower()
    if backend_id in STAGE1_VARIANTS:
        if kind == "mock":
            return MockNodeDetector(backend_id, spec, config)
        if backend_id == "E1" and kind in {"eupe"}:
            return EupeNodeDetector(backend_id, spec, config)
        if backend_id == "E1" and kind in {"grounding", "grounding-dino", "hf-grounding"}:
            return GroundingNodeDetector(backend_id, spec, config)
        if backend_id == "V1" and kind in {"qwen", "qwen-vl", "qwen3-vl", "transformers-qwen-vl"}:
            return QwenNodeDetector(backend_id, spec, config)
        return UnavailableNodeDetector(backend_id, spec, config)
    if backend_id in STAGE2_VARIANTS:
        if kind == "mock":
            return MockAttributeScorer(backend_id, spec, config)
        if backend_id == "E2" and kind in {"siglip", "siglip2", "hf-siglip"}:
            return SigLIPAttributeScorer(backend_id, spec, config)
        if backend_id == "V2" and kind in {"llava", "llava-next", "llava_next"}:
            return LlavaAttributeScorer(backend_id, spec, config)
        if backend_id == "V2" and kind in {"qwen", "qwen-vl", "qwen3-vl", "transformers-qwen-vl"}:
            return QwenAttributeScorer(backend_id, spec, config)
        return UnavailableAttributeScorer(backend_id, spec, config)
    if kind == "mock":
        return MockRelationScorer(backend_id, spec, config)
    if backend_id == "E3" and kind in {"siglip", "siglip2", "hf-siglip"}:
        return SigLIPRelationScorer(backend_id, spec, config)
    if backend_id == "V3" and kind in {"qwen", "qwen-vl", "qwen3-vl", "transformers-qwen-vl"}:
        return QwenRelationScorer(backend_id, spec, config)
    return UnavailableRelationScorer(backend_id, spec, config)


def build_pipeline_permutations() -> List[str]:
    return [f"{s1}-{s2}-{s3}" for s1 in STAGE1_VARIANTS for s2 in STAGE2_VARIANTS for s3 in STAGE3_VARIANTS]


def normalize_weights(weights: StageWeights) -> Dict[str, Dict[str, float]]:
    raw = {"node": weights.node, "attribute": weights.attribute, "relation": weights.relation}
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("Stage weights must sum to a positive value.")
    normalized = {key: value / total for key, value in raw.items()}
    return {"raw": raw, "normalized": normalized}


def normalize_active_weights(normalized_weights: Mapping[str, float], active_keys: Sequence[str]) -> Dict[str, float]:
    active_total = sum(normalized_weights[key] for key in active_keys)
    if active_total <= 0:
        return {key: 0.0 for key in active_keys}
    return {key: normalized_weights[key] / active_total for key in active_keys}


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def clamp_bbox(bbox: Sequence[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(x1 + 1, min(int(round(x2)), width))
    y2 = max(y1 + 1, min(int(round(y2)), height))
    return [x1, y1, x2, y2]


def normalized_bbox_to_pixel(bbox: Sequence[float], width: int, height: int) -> List[int]:
    if len(bbox) != 4:
        raise ValueError("Expected four bbox coordinates.")
    return clamp_bbox(
        [
            width * float(bbox[0]) / 1000.0,
            height * float(bbox[1]) / 1000.0,
            width * float(bbox[2]) / 1000.0,
            height * float(bbox[3]) / 1000.0,
        ],
        width,
        height,
    )


def parse_stage1_localization(raw_text: str, image_size: Tuple[int, int]) -> List[List[int]]:
    width, height = image_size
    parsed = json.loads(raw_text)
    boxes = parsed.get("boxes") if isinstance(parsed, dict) else parsed
    if isinstance(boxes, dict):
        boxes = boxes.get("boxes", [])
    if not isinstance(boxes, list):
        return []
    parsed_boxes = []
    for box in boxes:
        if isinstance(box, dict) and "bbox" in box:
            box = box["bbox"]
        if not isinstance(box, list) or len(box) != 4:
            continue
        parsed_boxes.append(normalized_bbox_to_pixel(box, width, height))
    return parsed_boxes


def prepare_square_crop(image: Image.Image, bbox: Sequence[int], size: int) -> Image.Image:
    cropped = image.crop(tuple(bbox))
    width, height = cropped.size
    side = max(width, height)
    canvas = Image.new(image.mode, (side, side))
    offset = ((side - width) // 2, (side - height) // 2)
    canvas.paste(cropped, offset)
    return canvas.resize((size, size))


def union_bbox(
    bbox_a: Sequence[int],
    bbox_b: Sequence[int],
    margin_ratio: float = 0.1,
    image_size: Optional[Tuple[int, int]] = None,
) -> List[int]:
    x1 = min(bbox_a[0], bbox_b[0])
    y1 = min(bbox_a[1], bbox_b[1])
    x2 = max(bbox_a[2], bbox_b[2])
    y2 = max(bbox_a[3], bbox_b[3])
    width = x2 - x1
    height = y2 - y1
    pad_x = width * margin_ratio
    pad_y = height * margin_ratio
    box = [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]
    if image_size is None:
        return [int(round(value)) for value in box]
    return clamp_bbox(box, image_size[0], image_size[1])


def draw_relation_markers(
    image: Image.Image,
    subject_bbox: Sequence[int],
    object_bbox: Sequence[int],
) -> Image.Image:
    array = np.array(image.convert("RGB"))
    cv2.rectangle(array, (subject_bbox[0], subject_bbox[1]), (subject_bbox[2], subject_bbox[3]), (255, 0, 0), 2)
    cv2.rectangle(array, (object_bbox[0], object_bbox[1]), (object_bbox[2], object_bbox[3]), (0, 0, 255), 2)
    return Image.fromarray(array)


def invert_relation(relation: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "subject": relation.get("object"),
        "relation": relation.get("relation"),
        "object": relation.get("subject"),
    }


def calibrate_score(raw_score: float, mode: str, scale: float, bias: float) -> float:
    score = raw_score * scale + bias
    if mode == "identity":
        return float(max(0.0, min(1.0, score)))
    if mode == "sigmoid":
        return float(1.0 / (1.0 + math.exp(-score)))
    if mode == "clip":
        return float(max(0.0, min(1.0, score)))
    raise ValueError(f"Unsupported calibration mode: {mode}")


def compose_score(stage_scores: Mapping[str, Optional[float]], normalized_weights: Mapping[str, float]) -> Dict[str, Any]:
    active_keys = [key for key, value in stage_scores.items() if value is not None]
    if not active_keys:
        return {"score": None, "active_weights": {}, "active_stage_scores": {}}
    active_weights = normalize_active_weights(normalized_weights, active_keys)
    score = sum(float(stage_scores[key]) * active_weights[key] for key in active_keys)
    return {
        "score": score,
        "active_weights": active_weights,
        "active_stage_scores": {key: stage_scores[key] for key in active_keys},
    }


def compute_prompt_aggregates(image_results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in image_results:
        grouped.setdefault(row["prompt_index"], []).append(row)

    prompt_rows = []
    for prompt_index, rows in sorted(grouped.items()):
        prompt_rows.append(
            {
                "prompt_index": prompt_index,
                "prompt": rows[0].get("prompt"),
                "image_count": len(rows),
                "average_final_score": safe_mean(row.get("final_score") for row in rows),
                "average_node_score": safe_mean(row["stage_scores"].get("node") for row in rows),
                "average_attribute_score": safe_mean(row["stage_scores"].get("attribute") for row in rows),
                "average_relation_score": safe_mean(row["stage_scores"].get("relation") for row in rows),
                "average_total_latency_ms": safe_mean(row["latency_ms"].get("total") for row in rows),
                "average_swap_delta": safe_mean(row.get("relation_swap", {}).get("delta_mean") for row in rows),
            }
        )
    return prompt_rows


def summarize_permutation(
    permutation: str,
    image_results: Sequence[Dict[str, Any]],
    prompt_rows: Sequence[Dict[str, Any]],
    normalized_weights: Mapping[str, float],
    raw_weights: Mapping[str, float],
    correlation: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    node_backend, attribute_backend, relation_backend = permutation.split("-")
    return {
        "permutation": permutation,
        "stage_backends": {
            "node": node_backend,
            "attribute": attribute_backend,
            "relation": relation_backend,
        },
        "image_count": len(image_results),
        "prompt_count": len(prompt_rows),
        "average_final_score": safe_mean(row.get("final_score") for row in image_results),
        "average_node_score": safe_mean(row["stage_scores"].get("node") for row in image_results),
        "average_attribute_score": safe_mean(row["stage_scores"].get("attribute") for row in image_results),
        "average_relation_score": safe_mean(row["stage_scores"].get("relation") for row in image_results),
        "average_total_latency_ms": safe_mean(row["latency_ms"].get("total") for row in image_results),
        "average_stage1_latency_ms": safe_mean(row["latency_ms"].get("stage1") for row in image_results),
        "average_stage2_latency_ms": safe_mean(row["latency_ms"].get("stage2") for row in image_results),
        "average_stage3_latency_ms": safe_mean(row["latency_ms"].get("stage3") for row in image_results),
        "relation_swap_delta_mean": safe_mean(
            row.get("relation_swap", {}).get("delta_mean") for row in image_results
        ),
        "swap_accuracy": safe_mean(
            row.get("relation_swap", {}).get("swap_accuracy") for row in image_results
        ),
        "raw_weights": dict(raw_weights),
        "normalized_weights": dict(normalized_weights),
        "correlation": correlation,
    }


def rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average_rank = (index + end - 1) / 2.0 + 1.0
        for pos in range(index, end):
            ranks[indexed[pos][0]] = average_rank
        index = end
    return ranks


def pearsonr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    x_std = x_arr.std()
    y_std = y_arr.std()
    if x_std == 0 or y_std == 0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearmanr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    return pearsonr(rankdata(xs), rankdata(ys))


def load_human_scores(label_config: LabelConfig) -> Optional[Dict[str, float]]:
    if not label_config.path:
        return None
    if not os.path.exists(label_config.path):
        return None
    rows = load_json_or_jsonl(label_config.path)
    scores = {}
    for row in rows:
        key = row.get(label_config.key_field)
        score = row.get(label_config.score_field)
        if key is None or score is None:
            continue
        scores[str(key)] = float(score)
    return scores


def extract_result_key(result: Mapping[str, Any], key_field: str) -> Optional[str]:
    if key_field in result:
        value = result.get(key_field)
    elif key_field == "prompt_index":
        value = result.get("prompt_index")
    elif key_field == "image_id":
        value = result.get("image_id")
    elif key_field == "prompt":
        value = result.get("prompt")
    else:
        value = None
    if value is None:
        return None
    return str(value)


def compute_correlation_report(
    image_results: Sequence[Dict[str, Any]],
    label_scores: Optional[Mapping[str, float]],
    label_config: LabelConfig,
) -> Optional[Dict[str, Any]]:
    if not label_scores:
        return None
    predicted = []
    labels = []
    missing = 0
    for row in image_results:
        result_key = extract_result_key(row, label_config.result_key_field)
        if result_key is None or result_key not in label_scores or row.get("final_score") is None:
            missing += 1
            continue
        predicted.append(float(row["final_score"]))
        labels.append(float(label_scores[result_key]))
    if len(predicted) < 2:
        return {
            "matched_count": len(predicted),
            "missing_count": missing,
            "pearson": None,
            "spearman": None,
        }
    return {
        "matched_count": len(predicted),
        "missing_count": missing,
        "pearson": pearsonr(predicted, labels),
        "spearman": spearmanr(predicted, labels),
    }


def load_experiment_items(config: ExperimentConfig) -> List[ExperimentItem]:
    if not config.prompts_file and not config.sg_file:
        raise ValueError("Provide either --prompts-file or --sg-file.")

    items: List[ExperimentItem] = []
    images_dir = Path(config.images_dir)

    if config.prompts_file:
        prompts_data = load_json_or_jsonl(config.prompts_file)
        end_idx = config.end_idx if config.end_idx is not None else len(prompts_data)
        stop_idx = min(end_idx, len(prompts_data))
        count = 0
        for idx in range(config.start_idx, stop_idx):
            if config.limit is not None and count >= config.limit:
                break
            entry = prompts_data[idx]
            scene_graph = extract_scene_graph(entry["meta_prompt"]["prompt"]) if "meta_prompt" in entry else None
            if not scene_graph or "error" in scene_graph:
                continue
            prompt = entry.get("prompt", "")
            for generation_idx in range(1, config.generation + 1):
                image_path = images_dir / config.image_pattern.format(index=idx, generation=generation_idx)
                if not image_path.exists():
                    continue
                items.append(
                    ExperimentItem(
                        prompt_index=idx,
                        image_id=f"{idx}:{generation_idx}",
                        prompt=prompt,
                        image_path=str(image_path),
                        scene_graph=scene_graph,
                        generation_index=generation_idx,
                    )
                )
            count += 1
        return items

    scene_graphs = load_json_or_jsonl(config.sg_file)
    end_idx = config.end_idx if config.end_idx is not None else len(scene_graphs)
    stop_idx = min(end_idx, len(scene_graphs))
    count = 0
    for idx in range(config.start_idx, stop_idx):
        if config.limit is not None and count >= config.limit:
            break
        scene_graph = scene_graphs[idx]
        filename = scene_graph.get("filename") or scene_graph.get("image_path") or scene_graph.get("image")
        if not filename:
            continue
        image_path = Path(filename)
        if not image_path.is_absolute():
            image_path = images_dir / filename
        if not image_path.exists():
            continue
        items.append(
            ExperimentItem(
                prompt_index=idx,
                image_id=str(scene_graph.get("id", image_path.stem)),
                prompt=scene_graph.get("prompt", image_path.stem),
                image_path=str(image_path),
                scene_graph=scene_graph,
            )
        )
        count += 1
    return items


def time_call(fn: Any, *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000.0


def run_ablation_experiment(
    config: ExperimentConfig,
    items: Optional[Sequence[ExperimentItem]] = None,
    backends: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    weight_info = normalize_weights(config.weights)
    normalized_weights = weight_info["normalized"]
    raw_weights = weight_info["raw"]
    experiment_items = list(items) if items is not None else load_experiment_items(config)
    backend_map = backends or {
        backend_id: build_backend(backend_id, config.backend_specs[backend_id], config)
        for backend_id in ("E1", "V1", "E2", "V2", "E3", "V3")
    }

    label_scores = load_human_scores(config.label_config)
    model_load_times = {
        key: getattr(backend, "model_load_time_ms", 0.0) for key, backend in backend_map.items()
    }
    image_rows_by_permutation: Dict[str, List[Dict[str, Any]]] = {name: [] for name in build_pipeline_permutations()}

    for item in experiment_items:
        with Image.open(item.image_path) as image_handle:
            image = image_handle.convert("RGB")

        stage1_cache: Dict[str, Dict[str, Any]] = {}
        for stage1_backend in STAGE1_VARIANTS:
            stage1_result, latency_ms = time_call(backend_map[stage1_backend].detect_nodes, image, item)
            stage1_cache[stage1_backend] = dict(stage1_result)
            stage1_cache[stage1_backend]["latency_ms"] = latency_ms

        stage2_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for stage1_backend in STAGE1_VARIANTS:
            for stage2_backend in STAGE2_VARIANTS:
                stage2_result, latency_ms = time_call(
                    backend_map[stage2_backend].score_attributes,
                    image,
                    item,
                    stage1_cache[stage1_backend],
                )
                stage2_cache[(stage1_backend, stage2_backend)] = dict(stage2_result)
                stage2_cache[(stage1_backend, stage2_backend)]["latency_ms"] = latency_ms

        stage3_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for stage1_backend in STAGE1_VARIANTS:
            for stage3_backend in STAGE3_VARIANTS:
                stage3_result, latency_ms = time_call(
                    backend_map[stage3_backend].score_relations,
                    image,
                    item,
                    stage1_cache[stage1_backend],
                )
                stage3_cache[(stage1_backend, stage3_backend)] = dict(stage3_result)
                stage3_cache[(stage1_backend, stage3_backend)]["latency_ms"] = latency_ms

        for permutation in build_pipeline_permutations():
            stage1_backend, stage2_backend, stage3_backend = permutation.split("-")
            stage1_result = stage1_cache[stage1_backend]
            stage2_result = stage2_cache[(stage1_backend, stage2_backend)]
            stage3_result = stage3_cache[(stage1_backend, stage3_backend)]
            stage_scores = {
                "node": stage1_result.get("fidelity_score"),
                "attribute": stage2_result.get("binding_score"),
                "relation": stage3_result.get("relation_score"),
            }
            composite = compose_score(stage_scores, normalized_weights)
            row = {
                "image_id": item.image_id,
                "prompt_index": item.prompt_index,
                "generation_index": item.generation_index,
                "prompt": item.prompt,
                "image_path": item.image_path,
                "permutation": permutation,
                "stage_scores": stage_scores,
                "final_score": composite["score"],
                "raw_weights": raw_weights,
                "normalized_weights": normalized_weights,
                "active_weights": composite["active_weights"],
                "latency_ms": {
                    "stage1": stage1_result["latency_ms"],
                    "stage2": stage2_result["latency_ms"],
                    "stage3": stage3_result["latency_ms"],
                    "total": stage1_result["latency_ms"] + stage2_result["latency_ms"] + stage3_result["latency_ms"],
                },
                "relation_swap": {
                    "delta_mean": stage3_result.get("swap_delta_mean"),
                    "swap_accuracy": stage3_result.get("swap_accuracy"),
                },
                "stages": {
                    "stage1": stage1_result,
                    "stage2": stage2_result,
                    "stage3": stage3_result,
                },
            }
            image_rows_by_permutation[permutation].append(row)

    permutation_reports = {}
    aggregate_rows = []
    latency_rows = []
    relation_rows = []

    for permutation, rows in image_rows_by_permutation.items():
        prompt_rows = compute_prompt_aggregates(rows)
        correlation = compute_correlation_report(rows, label_scores, config.label_config)
        summary = summarize_permutation(permutation, rows, prompt_rows, normalized_weights, raw_weights, correlation)
        permutation_reports[permutation] = {
            "permutation": permutation,
            "summary": summary,
            "prompt_aggregates": prompt_rows,
            "image_results": rows,
        }
        aggregate_rows.append(summary)
        latency_rows.append(
            {
                "permutation": permutation,
                "ms_per_image": summary["average_total_latency_ms"],
                "stage1_ms": summary["average_stage1_latency_ms"],
                "stage2_ms": summary["average_stage2_latency_ms"],
                "stage3_ms": summary["average_stage3_latency_ms"],
            }
        )
        relation_rows.append(
            {
                "permutation": permutation,
                "original_relation_score_mean": summary["average_relation_score"],
                "swapped_delta_mean": summary["relation_swap_delta_mean"],
                "swap_accuracy": summary["swap_accuracy"],
            }
        )

    correlation_report = {
        row["permutation"]: row["correlation"]
        for row in aggregate_rows
        if row.get("correlation") is not None
    }

    return {
        "config": serialize_config(config),
        "items_total": len(experiment_items),
        "permutations": permutation_reports,
        "aggregate_matrix": aggregate_rows,
        "latency_report": {
            "include_model_load_time": config.include_model_load_time,
            "model_load_time_ms": model_load_times if config.include_model_load_time else {},
            "rows": latency_rows,
        },
        "relation_swap_report": {"rows": relation_rows},
        "correlation_report": correlation_report if correlation_report else None,
    }


def serialize_config(config: ExperimentConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["weights"] = asdict(config.weights)
    payload["label_config"] = asdict(config.label_config)
    payload["backend_specs"] = {key: asdict(value) for key, value in config.backend_specs.items()}
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_experiment_outputs(report: Mapping[str, Any], output_dir: str) -> Dict[str, str]:
    root = Path(output_dir)
    permutations_dir = root / "permutations"
    artifact_paths = {
        "run_metadata": str(root / "run_metadata.json"),
        "aggregate_json": str(root / "aggregate_matrix.json"),
        "aggregate_csv": str(root / "aggregate_matrix.csv"),
        "latency_json": str(root / "latency_report.json"),
        "relation_json": str(root / "relation_swap_report.json"),
    }
    write_json(Path(artifact_paths["run_metadata"]), {"config": report["config"], "items_total": report["items_total"]})
    write_json(Path(artifact_paths["aggregate_json"]), report["aggregate_matrix"])
    write_csv(Path(artifact_paths["aggregate_csv"]), report["aggregate_matrix"])
    write_json(Path(artifact_paths["latency_json"]), report["latency_report"])
    write_json(Path(artifact_paths["relation_json"]), report["relation_swap_report"])

    if report.get("correlation_report") is not None:
        artifact_paths["correlation_json"] = str(root / "correlation_report.json")
        write_json(Path(artifact_paths["correlation_json"]), report["correlation_report"])

    for permutation, payload in report["permutations"].items():
        write_json(permutations_dir / f"{permutation}_details.json", payload)

    return artifact_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run 8-way graph-grounded alignment ablations.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--sg-file", default=None)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--image-pattern", default="{index:04d}-{generation}.png")
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--human-score-file", default=None)
    parser.add_argument("--label-key-field", default="image_id")
    parser.add_argument("--label-score-field", default="score")
    parser.add_argument("--result-key-field", default="image_id")
    parser.add_argument("--weight-node", type=float, default=0.3)
    parser.add_argument("--weight-attribute", type=float, default=0.3)
    parser.add_argument("--weight-relation", type=float, default=0.3)
    parser.add_argument("--node-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--node-nms-threshold", type=float, default=0.3)
    parser.add_argument("--stage2-crop-size", type=int, default=384)
    parser.add_argument("--stage2-calibration", default="clip", choices=["identity", "clip", "sigmoid"])
    parser.add_argument("--stage2-calibration-scale", type=float, default=1.0)
    parser.add_argument("--stage2-calibration-bias", type=float, default=0.0)
    parser.add_argument("--stage3-margin-ratio", type=float, default=0.1)
    parser.add_argument("--include-model-load-time", action="store_true")
    parser.add_argument("--e1-backend-kind", default="mock")
    parser.add_argument("--v1-backend-kind", default="mock")
    parser.add_argument("--e2-backend-kind", default="mock")
    parser.add_argument("--v2-backend-kind", default="mock")
    parser.add_argument("--e3-backend-kind", default="mock")
    parser.add_argument("--v3-backend-kind", default="mock")
    parser.add_argument("--eupe-model-path", default=None)
    parser.add_argument("--qwen-model-path", default=None)
    parser.add_argument("--siglip-model-path", default=None)
    parser.add_argument("--llava-model-path", default=None)
    parser.add_argument("--eupe-checkpoint-path", default=None)
    parser.add_argument("--qwen-checkpoint-path", default=None)
    parser.add_argument("--siglip-checkpoint-path", default=None)
    parser.add_argument("--llava-checkpoint-path", default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        output_dir=args.output_dir,
        prompts_file=None if args.prompts_file in {None, "None"} else args.prompts_file,
        sg_file=None if args.sg_file in {None, "None"} else args.sg_file,
        images_dir=args.images_dir,
        image_pattern=args.image_pattern,
        generation=args.generation,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        limit=args.limit,
        weights=StageWeights(args.weight_node, args.weight_attribute, args.weight_relation),
        node_confidence_threshold=args.node_confidence_threshold,
        node_nms_threshold=args.node_nms_threshold,
        stage2_crop_size=args.stage2_crop_size,
        stage2_calibration=args.stage2_calibration,
        stage2_calibration_scale=args.stage2_calibration_scale,
        stage2_calibration_bias=args.stage2_calibration_bias,
        stage3_margin_ratio=args.stage3_margin_ratio,
        include_model_load_time=args.include_model_load_time,
        label_config=LabelConfig(
            path=None if args.human_score_file in {None, "None"} else args.human_score_file,
            key_field=args.label_key_field,
            score_field=args.label_score_field,
            result_key_field=args.result_key_field,
        ),
        backend_specs={
            "E1": BackendSpec(args.e1_backend_kind, args.eupe_model_path, args.eupe_checkpoint_path),
            "V1": BackendSpec(args.v1_backend_kind, args.qwen_model_path, args.qwen_checkpoint_path),
            "E2": BackendSpec(args.e2_backend_kind, args.siglip_model_path, args.siglip_checkpoint_path),
            "V2": BackendSpec(args.v2_backend_kind, args.llava_model_path, args.llava_checkpoint_path),
            "E3": BackendSpec(args.e3_backend_kind, args.siglip_model_path, args.siglip_checkpoint_path),
            "V3": BackendSpec(args.v3_backend_kind, args.qwen_model_path, args.qwen_checkpoint_path),
        },
    )
