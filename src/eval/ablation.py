"""Experiment harness for graph-grounded alignment ablations."""

from __future__ import annotations

import argparse, csv, hashlib, json, math, os, time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

import os
import re

from modules.processing import (
    apply_batch_results,
    build_attribute_prompt,
    build_object_prompt,
    build_relation_prompt,
    image_path_from_pattern,
    load_json_or_jsonl,
    summarize_results,
    extract_scene_graph
)

from tqdm import tqdm

STAGE1_VARIANTS, STAGE2_VARIANTS, STAGE3_VARIANTS = ("E1", "V1"), ("E2", "V2"), ("E3", "V3")

# def extract_scene_graph(prompt_text: str) -> Dict[str, Any]:
#     text = prompt_text.split("Current Task:")[-1] if "Current Task:" in prompt_text else prompt_text
#     try:
#         obj_sec = text.split("Objects:", 1)[1].split("Relationships:", 1)[0]
#         rel_sec = text.split("Relationships:", 1)[1].split("[Step-by-Step Reasoning]", 1)[0]
#     except IndexError:
#         return {"error": "Could not find expected Objects or Relationships sections."}

#     objects, relations, current_obj = [], [], None
#     for line in obj_sec.strip().splitlines():
#         line = line.strip()
#         if not line: continue
#         if line.startswith("-") and "(object id" in line:
#             name = line[1:].split("(object id", 1)[0].strip().split(" ", 1)[-1].strip()
#             obj_id = int(line.split(":")[-1].split(")")[0].strip())
#             current_obj = {"id": obj_id, "name": name, "attributes": []}
#             objects.append(current_obj)
#         elif current_obj and line.startswith("-"):
#             current_obj["attributes"].append(line[1:].strip())

#     for line in rel_sec.strip().splitlines():
#         tokens = line.strip()[1:].strip().split()
#         if line.strip().startswith("- Object") and len(tokens) >= 5:
#             relations.append({"subject": int(tokens[1]), "relation": " ".join(tokens[2:-2]), "object": int(tokens[-1])})

#     return {"objects": objects, "relations": relations}

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    return json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]

def resolve_siglip_model_path(model_path: str) -> str:
    path = Path(model_path)
    if path.is_dir() and (path / "config.json").exists(): return str(path)
    if path.is_dir() and any(path.glob("*.npz")):
        sibling = path.parent / path.name.replace("-jax", "")
        if sibling.exists() and (sibling / "config.json").exists(): return str(sibling)
        raise ValueError(f"Requires Transformers-compatible checkpoint, found JAX weights at {model_path}.")
    return model_path

@dataclass(frozen=True)
class StageWeights: node: float; attribute: float; relation: float

@dataclass(frozen=True)
class ExperimentItem:
    prompt_index: int; image_id: str; prompt: str; image_path: str; scene_graph: Dict[str, Any]; generation_index: Optional[int] = None

@dataclass(frozen=True)
class LabelConfig: path: Optional[str]; key_field: str = "image_id"; score_field: str = "score"; result_key_field: str = "image_id"

@dataclass(frozen=True)
class BackendSpec: kind: str; model_path: Optional[str] = None; checkpoint_path: Optional[str] = None

@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: str; prompts_file: Optional[str]; sg_file: Optional[str]; images_dir: str; image_pattern: str; generation: int
    start_idx: int; end_idx: Optional[int]; limit: Optional[int]; weights: StageWeights
    node_confidence_threshold: float; node_nms_threshold: float; stage2_crop_size: int
    stage2_calibration: str; stage2_calibration_scale: float; stage2_calibration_bias: float; stage3_margin_ratio: float
    include_model_load_time: bool; label_config: LabelConfig; backend_specs: Dict[str, BackendSpec]
    selected_backends: Optional[set] = None; use_cpu: bool = False; low_vram: bool = False; use_vllm: bool = False
    max_text_length: int = 64; torch_cuda_mem_frac: float = 0.8
    vllm_api_base: str = "http://127.0.0.1:8000/v1"; vllm_api_key: Optional[str] = None
    vllm_temperature: Optional[float] = None; vllm_max_tokens: Optional[int] = None; vllm_yes_no_max_tokens: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, 'selected_backends', self.selected_backends or None)

class StageBackend(ABC):
    def __init__(self, backend_id: str, spec: BackendSpec, config: ExperimentConfig):
        self.backend_id, self.spec, self.config, self.model_load_time_ms = backend_id, spec, config, 0.0

def _default_qwen_model_path() -> str:
    return "/fs/nexus-projects/scene_graph_sd/Qwen3-VL-8B-Instruct"

def _default_siglip_model_path() -> str:
    return "google/siglip2-so400m-patch14-384"

def _backend_runtime_key(backend_id: str, spec: BackendSpec, config: ExperimentConfig) -> Optional[Tuple[Any, ...]]:
    kind = (spec.kind or "").lower()

    if backend_id in {"E2", "E3"} and "siglip" in kind:
        model_path = resolve_siglip_model_path(spec.model_path) if spec.model_path else _default_siglip_model_path()
        return ("siglip", model_path, spec.checkpoint_path, config.use_cpu)

    if backend_id in {"V1", "V2", "V3"} and "qwen" in kind:
        model_path = spec.model_path or _default_qwen_model_path()
        runtime_kind = "qwen-vllm" if config.use_vllm else "qwen-hf"
        if config.use_vllm:
            return (runtime_kind, model_path, config.vllm_api_base)
        return (runtime_kind, model_path, spec.checkpoint_path, config.use_cpu)

    return None

class NodeDetectorBackend(StageBackend):
    @abstractmethod
    def detect_nodes(self, image: Image.Image, item: ExperimentItem) -> Dict[str, Any]: raise NotImplementedError

class AttributeScorerBackend(StageBackend):
    def score_attributes(self, image: Image.Image, item: ExperimentItem, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        node_map = {n["id"]: n for n in stage1_result.get("nodes", [])}
        results = []
        for entity in item.scene_graph.get("objects", []):
            for attr in entity.get("attributes", []):
                node = node_map.get(entity.get("id"))
                if not node or not node.get("passed") or not node.get("bbox"):
                    results.append({"id": entity.get("id"), "name": entity.get("name"), "attribute": attr, "skipped": True, "skip_reason": "node_not_localized"})
                    continue
                crop = prepare_square_crop(image, node["bbox"], self.config.stage2_crop_size)
                raw_score, extra = self._evaluate_attribute(crop, entity.get("name"), attr)
                cal_score = calibrate_score(raw_score, self.config.stage2_calibration, self.config.stage2_calibration_scale, self.config.stage2_calibration_bias)
                results.append({"id": entity.get("id"), "name": entity.get("name"), "attribute": attr, "score": raw_score, "calibrated_score": cal_score, "skipped": False, "bbox": node["bbox"], **extra})
        return {"backend": self.backend_id, "crop_size": self.config.stage2_crop_size, "attributes": results, "binding_score": safe_mean(e.get("calibrated_score") for e in results if not e.get("skipped"))}

    @abstractmethod
    def _evaluate_attribute(self, crop: Image.Image, entity_name: str, attribute: str) -> Tuple[float, dict]: raise NotImplementedError

class RelationScorerBackend(StageBackend):
    def score_relations(self, image: Image.Image, item: ExperimentItem, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        node_map = {n["id"]: n for n in stage1_result.get("nodes", [])}
        entity_map = {e["id"]: e for e in item.scene_graph.get("objects", [])}
        results = []
        for rel in item.scene_graph.get("relations", []):
            subj, obj = node_map.get(rel.get("subject")), node_map.get(rel.get("object"))
            if not subj or not obj or not subj.get("bbox") or not obj.get("bbox"):
                results.append({"subject": rel.get("subject"), "relation": rel.get("relation"), "object": rel.get("object"), "skipped": True, "skip_reason": "missing_localization"})
                continue
            s_name, o_name = entity_map.get(rel["subject"], {}).get("name", "sub"), entity_map.get(rel["object"], {}).get("name", "obj")
            orig_raw, swap_raw, extra = self._evaluate_relation(image, subj["bbox"], obj["bbox"], rel["relation"], s_name, o_name)
            
            cal = lambda s: calibrate_score(s, self.config.stage2_calibration, self.config.stage2_calibration_scale, self.config.stage2_calibration_bias)
            orig_score, swap_score = cal(orig_raw), cal(swap_raw)
            results.append({"subject": rel["subject"], "relation": rel["relation"], "object": rel["object"], "original_score": orig_score, "swapped_score": swap_score, "delta": orig_score - swap_score, "swap_correct": orig_score > swap_score, "skipped": False, **extra})
        
        return {"backend": self.backend_id, "relations": results, "relation_score": safe_mean(e.get("original_score") for e in results if not e.get("skipped")), "swap_accuracy": safe_mean(1.0 if e.get("swap_correct") else 0.0 for e in results if e.get("swap_correct") is not None)}

    @abstractmethod
    def _evaluate_relation(self, image: Image.Image, subj_bbox: list, obj_bbox: list, relation: str, subj_name: str, obj_name: str) -> Tuple[float, float, dict]: raise NotImplementedError

class _TransformersBackendMixin:
    def _load_components(self) -> Any:
        from transformers import AutoModel, AutoModelForCausalLM, AutoModelForZeroShotObjectDetection, AutoProcessor
        return AutoProcessor, AutoModel, AutoModelForCausalLM, AutoModelForZeroShotObjectDetection

    def _load_model(self, model_loader: Any, **extra) -> Tuple[Any, Any]:
        import torch
        AutoProc, *_ = self._load_components()
        start = time.perf_counter()
        proc = AutoProc.from_pretrained(self.spec.model_path, trust_remote_code=True)
        kwargs = {"trust_remote_code": True, "torch_dtype": "auto", "device_map": "auto" if torch.cuda.is_available() else None, **extra}
        if kwargs.get("device_map") is None: kwargs.pop("device_map")
        model = model_loader.from_pretrained(self.spec.model_path, **kwargs)
        if self.spec.checkpoint_path:
            ckpt = torch.load(self.spec.checkpoint_path, map_location="cpu")
            model.load_state_dict({k.replace("module.", "", 1): v for k, v in (ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt).items()}, strict=False)
        model.eval()
        self.model_load_time_ms = (time.perf_counter() - start) * 1000.0
        return proc, model

    def _to_device(self, batch: dict, model: Any) -> dict:
        device = getattr(model, "device", next(model.parameters()).device)
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


# --- UPDATED COMPONENT FOR STAGE 1: DINO + CLIP ---
class DinoClipNodeDetector(NodeDetectorBackend, _TransformersBackendMixin):
    def __init__(self, *args):
        super().__init__(*args)
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, CLIPProcessor, CLIPModel

        start = time.perf_counter()

        # Determine device: respect --use-cpu flag, otherwise use auto placement
        cuda_available = torch.cuda.is_available() and not self.config.use_cpu
        device_map = "auto" if cuda_available else None

        # Load DINO for box proposals (falls back to generic DINO if no model_path specified)
        dino_path = self.spec.model_path or "IDEA-Research/grounding-dino-base"
        self.dino_proc = AutoProcessor.from_pretrained(dino_path)
        # GroundingDINO is sensitive to dtype - use float32 for stability
        dino_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float32}
        if device_map:
            dino_kwargs["device_map"] = device_map
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_path, **dino_kwargs)

        # Load CLIP for text/image matching
        clip_path = "openai/clip-vit-base-patch32"
        self.clip_proc = CLIPProcessor.from_pretrained(clip_path)
        clip_kwargs = {"torch_dtype": torch.float16 if cuda_available else torch.float32}
        if device_map:
            clip_kwargs["device_map"] = device_map
        self.clip_model = CLIPModel.from_pretrained(clip_path, **clip_kwargs)

        # Infer device from model placement
        self.device = getattr(self.dino_model, "device", None) or ("cuda" if cuda_available else "cpu")
        self.dino_model.eval()
        self.clip_model.eval()

        self.model_load_time_ms = (time.perf_counter() - start) * 1000.0

    def detect_nodes(self, image: Image.Image, item: ExperimentItem) -> Dict[str, Any]:
        import torch
        # 1. Propose bounding boxes using DINO (using a generic 'object' prompt to capture proposals)
        dino_inputs = self.dino_proc(images=image, text="object .", return_tensors="pt").to(self.device)
        with torch.inference_mode():
            dino_outputs = self.dino_model(**dino_inputs)
            
        target_sizes = torch.tensor([[image.size[1], image.size[0]]], device=self.device)
        results = self.dino_proc.post_process_grounded_object_detection(
            dino_outputs,
            dino_inputs.input_ids,
            threshold=0.1, 
            # box_threshold=0.1,  # Lower threshold to recall all possible objects
            text_threshold=0.1,
            target_sizes=target_sizes
        )[0]
        
        boxes = results["boxes"].cpu().tolist()
        
        # 2. Extract crops for valid boxes
        crops = []
        valid_boxes = []
        for box in boxes:
            clamped = clamp_bbox(box, *image.size)
            if clamped[2] > clamped[0] and clamped[3] > clamped[1]:
                crops.append(image.crop(tuple(clamped)))
                valid_boxes.append(clamped)
                
        nodes = []
        if not crops:
            # Fallback if no proposals are generated at all
            for entity in item.scene_graph.get("objects", []):
                nodes.append({"id": entity.get("id"), "name": entity.get("name"), "bbox": None, "confidence": 0.0, "passed": False, "score": 0.0})
            return {"backend": self.backend_id, "nodes": nodes, "fidelity_score": 0.0}

        # 3. Compute CLIP similarities to match proposed crops with node labels
        labels = [e.get("name", "") for e in item.scene_graph.get("objects", [])]
        clip_texts = [f"a photo of a {label}" for label in labels]
        print("Number of clip texts: ", len(clip_texts))
        clip_inputs = self.clip_proc(text=clip_texts, images=crops, return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            clip_outputs = self.clip_model(**clip_inputs)
            # logits_per_text shape: (num_labels, num_crops)
            logits_per_text = clip_outputs.logits_per_text
            
            # Softmax to find which crop best matches each individual label
            probs = logits_per_text.softmax(dim=-1)

        # 4. Assign the best bounding box to each object based on CLIP
        for i, entity in enumerate(item.scene_graph.get("objects", [])):
            best_idx = torch.argmax(logits_per_text[i]).item()
            best_conf = probs[i][best_idx].item()
            best_box = valid_boxes[best_idx]
            
            passed = bool(best_box and best_conf >= self.config.node_confidence_threshold)
            nodes.append({
                "id": entity.get("id"),
                "name": entity.get("name"),
                "bbox": best_box if passed else None,
                "confidence": best_conf,
                "passed": passed,
                "score": 1.0 if passed else 0.0
            })
            
        return {"backend": self.backend_id, "nodes": nodes, "fidelity_score": safe_mean(n["score"] for n in nodes)}
# --- END UPDATED COMPONENT ---


class _VisionLanguageMixin(_TransformersBackendMixin):
    @classmethod
    def load_shared_runtime(cls, spec: BackendSpec, config: ExperimentConfig) -> Tuple[Dict[str, Any], float]:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        # Respect --use-cpu flag
        cuda_available = torch.cuda.is_available() and not config.use_cpu
        start = time.perf_counter()

        # Qwen3-VL requires using Qwen3VLForConditionalGeneration directly since Qwen3VLConfig
        # is not registered with AutoModel or AutoModelForCausalLM
        model_path = spec.model_path or _default_qwen_model_path()
        proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # Optimizations for faster inference:
        # - flash_attention_2: 2-3x faster attention
        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if cuda_available else torch.float32,
            "attn_implementation": "flash_attention_2" if cuda_available else "eager",
        }
        if cuda_available:
            kwargs["device_map"] = "auto"
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
        if spec.checkpoint_path:
            ckpt = torch.load(spec.checkpoint_path, map_location="cpu")
            model.load_state_dict({k.replace("module.", "", 1): v for k, v in (ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt).items()}, strict=False)
        model.eval()
        load_time_ms = (time.perf_counter() - start) * 1000.0

        return ({
            "proc": proc,
            "model": model,
            "_cuda_available": cuda_available,
            "_batch_size": 1,
            "_max_new_tokens": 4096,
        }, load_time_ms)

    def __init__(self, *args, shared_runtime: Optional[Dict[str, Any]] = None):
        super().__init__(*args)
        runtime = shared_runtime
        if runtime is None:
            runtime, self.model_load_time_ms = self.load_shared_runtime(self.spec, self.config)
        self.proc = runtime["proc"]
        self.model = runtime["model"]
        self._cuda_available = runtime["_cuda_available"]
        self._batch_size = runtime["_batch_size"]
        self._max_new_tokens = runtime["_max_new_tokens"]

    def generate_text(self, image: Image.Image, prompt: str) -> str:
        import torch
        batch = self._to_device(self.proc.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"), self.model)
        with torch.inference_mode(): gen = self.model.generate(**batch, max_new_tokens=self._max_new_tokens, do_sample=False)
        return self.proc.batch_decode(gen[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def yes_no_score(self, image: Image.Image, prompt: str) -> Tuple[float, dict]:
        import torch
        batch = self._to_device(self.proc.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"), self.model)
        with torch.inference_mode(): out = self.model.generate(**batch, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
        logits = out.scores[0][0]
        y, n = self.proc.tokenizer.encode("Yes", add_special_tokens=False)[0], self.proc.tokenizer.encode("No", add_special_tokens=False)[0]
        probs = torch.softmax(torch.stack([logits[y], logits[n]]), dim=0).tolist()
        return probs[0], {"yes_prob": probs[0], "no_prob": probs[1]}

class QwenNodeDetector( _VisionLanguageMixin, NodeDetectorBackend):
    """Optimized Qwen node detector that batches all entities per image into one prompt."""

    def detect_nodes(self, image: Image.Image, item: ExperimentItem) -> Dict[str, Any]:
        objects = item.scene_graph.get("objects", [])
        if not objects:
            return {"backend": self.backend_id, "nodes": [], "fidelity_score": 1.0}

        # Build a single prompt asking for all entities (reduces N queries to 1)
        entity_list = "\n".join(f"  - Entity {i+1}: {obj.get('name')} (id: {obj.get('id')})" for i, obj in enumerate(objects))
        prompt = f"""Analyze the image and locate all the following entities. Return a JSON object with this exact structure:
{{
  "results": [
    {{"entity_id": <id>, "name": "<name>", "boxes": [[<x1>, <y1>, <x2>, <y2>], ...], "confidence": <0-1>}},
    ...
  ]
}}

For each entity, provide bounding boxes in normalized coordinates [0,1000]. If an entity is not found, use empty boxes [] and confidence 0.0.

Entities to locate:
{entity_list}

Return ONLY the JSON object, nothing else."""

        raw = self.generate_text(image, prompt)
        print("VLM Stage 1 output:", raw)

        # Parse the batched JSON response
        nodes = []
        try:
            parsed = json.loads(raw[raw.find("{"):raw.rfind("}")+1]) if "{" in raw else {}
            results = parsed.get("results", [])
            for result in results:
                entity_id = result.get("entity_id")
                name = result.get("name")
                boxes_raw = result.get("boxes", [])
                conf = float(result.get("confidence", 0.0))
                boxes = parse_stage1_localization(boxes_raw, image.size)
                passed = bool(boxes) and conf >= self.config.node_confidence_threshold
                nodes.append({
                    "id": entity_id,
                    "name": name,
                    "bbox": boxes[0] if passed else None,
                    "confidence": conf,
                    "passed": passed,
                    "score": 1.0 if passed else 0.0
                })
        except json.JSONDecodeError:
            # On parse failure, return empty results
            pass

        return {"backend": self.backend_id, "nodes": nodes, "fidelity_score": safe_mean(n["score"] for n in nodes)}

class _QwenVLLMMixin:
    @classmethod
    def load_shared_runtime(cls, spec: BackendSpec, config: ExperimentConfig) -> Tuple[Dict[str, Any], float]:
        import os
        import urllib.request

        model_path = spec.model_path or _default_qwen_model_path()
        api_base = config.vllm_api_base.rstrip("/")
        api_key = config.vllm_api_key if config.vllm_api_key is not None else os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            request = urllib.request.Request(f"{api_base}/models", headers=headers)
            with urllib.request.urlopen(request, timeout=10) as response:
                models_payload = json.loads(response.read().decode("utf-8"))
            served_model_ids = [m.get("id") for m in models_payload.get("data", []) if m.get("id")]
            if model_path not in served_model_ids and len(served_model_ids) == 1:
                print(f"Qwen vLLM runtime: server model id is {served_model_ids[0]!r}; overriding requested model {model_path!r}")
                model_path = served_model_ids[0]
        except Exception as exc:
            print(f"Qwen vLLM runtime: warning: could not query {api_base}/models ({exc}); using requested model {model_path!r}")
        print(f"Qwen vLLM runtime: using OpenAI-compatible server {api_base} with model {model_path}")

        start = time.perf_counter()
        sampling_params = {}
        yes_no_sampling_params = {}
        if config.vllm_temperature is not None:
            sampling_params["temperature"] = config.vllm_temperature
            yes_no_sampling_params["temperature"] = config.vllm_temperature
        if config.vllm_max_tokens is not None:
            sampling_params["max_tokens"] = config.vllm_max_tokens
        if config.vllm_yes_no_max_tokens is not None:
            yes_no_sampling_params["max_tokens"] = config.vllm_yes_no_max_tokens
        load_time_ms = (time.perf_counter() - start) * 1000.0
        return ({
            "llm": {"api_base": api_base, "model": model_path},
            "api_base": api_base,
            "api_key": api_key,
            "model": model_path,
            "sampling_params": sampling_params,
            "yes_no_sampling_params": yes_no_sampling_params,
            "timeout_s": 300,
        }, load_time_ms)

    def __init__(self, backend_id: str, spec: BackendSpec, config: ExperimentConfig, shared_runtime: Optional[Dict[str, Any]] = None):
        super().__init__(backend_id, spec, config)
        runtime = shared_runtime
        if runtime is None:
            runtime, self.model_load_time_ms = self.load_shared_runtime(spec, config)
        self.llm = runtime.get("llm", runtime)
        self.api_base = runtime.get("api_base")
        self.api_key = runtime.get("api_key")
        self.model = runtime.get("model", spec.model_path or _default_qwen_model_path())
        self.timeout_s = runtime.get("timeout_s", 300)
        self.sampling_params = runtime["sampling_params"]
        self.yes_no_sampling_params = runtime.get("yes_no_sampling_params", self.sampling_params)

class QwenNodeDetectorVLLM(_QwenVLLMMixin, NodeDetectorBackend):
    """Qwen node detector using vLLM for faster inference with batching."""

    def _image_to_base64(self, image: Image.Image) -> str:
        import io, base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _messages_for_image_url(self, image_url: str, prompt: str) -> List[Dict[str, Any]]:
        return [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        }]

    def _messages_for_image(self, image: Image.Image, prompt: str) -> List[Dict[str, Any]]:
        image_url = f"data:image/png;base64,{self._image_to_base64(image)}"
        return self._messages_for_image_url(image_url, prompt)

    def _chat_texts(self, conversations: List[List[Dict[str, Any]]], sampling_params: Any) -> List[str]:
        if not conversations:
            return []
        from concurrent.futures import ThreadPoolExecutor

        max_workers = min(16, len(conversations))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(lambda messages: self._chat_text(messages, sampling_params), conversations))

    def _chat_text(self, messages: List[Dict[str, Any]], sampling_params: Mapping[str, Any]) -> str:
        import urllib.error
        import urllib.request

        if not self.api_base:
            # Compatibility path for tests that monkeypatch a fake in-process runtime.
            response = self.llm.chat([messages], sampling_params=sampling_params)
            return response[0].outputs[0].text.strip()

        payload = {
            "model": self.model,
            "messages": messages,
            **dict(sampling_params),
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM server request failed with HTTP {exc.code}: {detail}") from exc

        return body["choices"][0]["message"]["content"].strip()

    def detect_nodes(self, image: Image.Image, item: ExperimentItem) -> Dict[str, Any]:
        """Process all entities in a single prompt per image for efficiency."""
        objects = item.scene_graph.get("objects", [])
        if not objects:
            return {"backend": self.backend_id, "nodes": [], "fidelity_score": 1.0}

        # Build a single prompt asking for all entities
        entity_list = "\n".join(f"  - Entity {i+1}: {obj.get('name')} (id: {obj.get('id')})" for i, obj in enumerate(objects))
        prompt = f"""Analyze the image and locate all the following entities. Return a JSON object with this exact structure:
{{
  "results": [
    {{"entity_id": <id>, "name": "<name>", "boxes": [[<x1>, <y1>, <x2>, <y2>], ...], "confidence": <0-1>}},
    ...
  ]
}}

For each entity, provide bounding boxes in normalized coordinates [0,1000]. If an entity is not found, use empty boxes [] and confidence 0.0.

Entities to locate:
{entity_list}

Return ONLY the JSON object, nothing else."""

        raw = self._chat_texts([self._messages_for_image(image, prompt)], self.sampling_params)[0]

        # Parse the JSON response
        nodes = []
        try:
            parsed = json.loads(raw[raw.find("{"):raw.rfind("}")+1]) if "{" in raw else {}
            results = parsed.get("results", [])
            for result in results:
                entity_id = result.get("entity_id")
                name = result.get("name")
                boxes_raw = result.get("boxes", [])
                conf = float(result.get("confidence", 0.0))
                boxes = parse_stage1_localization(boxes_raw, image.size)
                passed = bool(boxes) and conf >= self.config.node_confidence_threshold
                nodes.append({
                    "id": entity_id,
                    "name": name,
                    "bbox": boxes[0] if passed else None,
                    "confidence": conf,
                    "passed": passed,
                    "score": 1.0 if passed else 0.0
                })
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON for image {item.image_id}: {e}")
            print(f"Raw response: {raw[:200]}...")
            # Return empty results on parse failure
            pass

        return {"backend": self.backend_id, "nodes": nodes, "fidelity_score": safe_mean(n["score"] for n in nodes)}

    def yes_no_score(self, image: Image.Image, prompt: str) -> Tuple[float, dict]:
        score, probs, text = self.batch_yes_no_score([(image, prompt)])[0]
        text = text.strip().lower()
        if text.startswith("yes") or text.startswith("no"):
            return score, probs
        yes_prob = 1.0 if text.startswith("yes") else 0.0
        return yes_prob, {"yes_prob": yes_prob, "no_prob": 1.0 - yes_prob}

    def batch_yes_no_score(self, requests: List[Tuple[Image.Image, str]]) -> List[Tuple[float, Dict[str, float], str]]:
        conversations = [self._messages_for_image(image, prompt) for image, prompt in requests]
        texts = self._chat_texts(conversations, self.yes_no_sampling_params)
        results = []
        for text in texts:
            yes_prob = 1.0 if text.strip().lower().startswith("yes") else 0.0
            results.append((yes_prob, {"yes_prob": yes_prob, "no_prob": 1.0 - yes_prob}, text))
        return results

class QwenAttributeClassifierVLLM(QwenNodeDetectorVLLM, AttributeScorerBackend):
    def score_attributes(self, image: Image.Image, item: ExperimentItem, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        node_map = {n["id"]: n for n in stage1_result.get("nodes", [])}
        results = []
        requests = []
        request_indices = []

        for entity in item.scene_graph.get("objects", []):
            for attr in entity.get("attributes", []):
                node = node_map.get(entity.get("id"))
                if not node or not node.get("passed") or not node.get("bbox"):
                    results.append({"id": entity.get("id"), "name": entity.get("name"), "attribute": attr, "skipped": True, "skip_reason": "node_not_localized"})
                    continue

                crop = prepare_square_crop(image, node["bbox"], self.config.stage2_crop_size)
                prompt = f"Answer strictly with Yes or No.\nDoes this crop show a {entity.get('name')} with attribute '{attr}'?"
                request_indices.append(len(results))
                requests.append((crop, prompt))
                results.append({"id": entity.get("id"), "name": entity.get("name"), "attribute": attr, "skipped": False, "bbox": node["bbox"]})

        for result_idx, (raw_score, probs, _) in zip(request_indices, self.batch_yes_no_score(requests)):
            cal_score = calibrate_score(raw_score, self.config.stage2_calibration, self.config.stage2_calibration_scale, self.config.stage2_calibration_bias)
            results[result_idx].update({"score": raw_score, "calibrated_score": cal_score, "token_probs": probs})

        return {"backend": self.backend_id, "crop_size": self.config.stage2_crop_size, "attributes": results, "binding_score": safe_mean(e.get("calibrated_score") for e in results if not e.get("skipped"))}

    def _evaluate_attribute(self, crop, entity_name, attribute):
        score, probs = self.yes_no_score(crop, f"Answer strictly with Yes or No.\nDoes this crop show a {entity_name} with attribute '{attribute}'?")
        return score, {"token_probs": probs}

class QwenRelationScorerVLLM(QwenNodeDetectorVLLM, RelationScorerBackend):
    def score_relations(self, image: Image.Image, item: ExperimentItem, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        node_map = {n["id"]: n for n in stage1_result.get("nodes", [])}
        entity_map = {e["id"]: e for e in item.scene_graph.get("objects", [])}
        results = []
        requests = []
        request_indices = []
        prompt_template = "Answer strictly with Yes or No.\nThe red box marks the subject and the blue box marks the object.\nIs the relation true: {s} {r} {o}?"

        for rel in item.scene_graph.get("relations", []):
            subj, obj = node_map.get(rel.get("subject")), node_map.get(rel.get("object"))
            if not subj or not obj or not subj.get("bbox") or not obj.get("bbox"):
                results.append({"subject": rel.get("subject"), "relation": rel.get("relation"), "object": rel.get("object"), "skipped": True, "skip_reason": "missing_localization"})
                continue

            s_name = entity_map.get(rel["subject"], {}).get("name", "sub")
            o_name = entity_map.get(rel["object"], {}).get("name", "obj")
            marked = draw_relation_markers(image, subj["bbox"], obj["bbox"])
            request_indices.append(len(results))
            requests.append((marked, prompt_template.format(s=s_name, r=rel["relation"], o=o_name)))
            requests.append((marked, prompt_template.format(s=o_name, r=rel["relation"], o=s_name)))
            results.append({"subject": rel["subject"], "relation": rel["relation"], "object": rel["object"], "skipped": False, "marker_mode": marked.mode})

        yes_no_results = self.batch_yes_no_score(requests)
        for result_idx, pair_start in zip(request_indices, range(0, len(yes_no_results), 2)):
            orig_raw, orig_p, _ = yes_no_results[pair_start]
            swap_raw, swap_p, _ = yes_no_results[pair_start + 1]
            cal = lambda s: calibrate_score(s, self.config.stage2_calibration, self.config.stage2_calibration_scale, self.config.stage2_calibration_bias)
            orig_score, swap_score = cal(orig_raw), cal(swap_raw)
            results[result_idx].update({
                "original_score": orig_score,
                "swapped_score": swap_score,
                "delta": orig_score - swap_score,
                "swap_correct": orig_score > swap_score,
                "token_probs": {"original": orig_p, "swapped": swap_p},
            })

        return {"backend": self.backend_id, "relations": results, "relation_score": safe_mean(e.get("original_score") for e in results if not e.get("skipped")), "swap_accuracy": safe_mean(1.0 if e.get("swap_correct") else 0.0 for e in results if e.get("swap_correct") is not None)}

    def _evaluate_relation(self, image, subj_bbox, obj_bbox, relation, subj_name, obj_name):
        marked = draw_relation_markers(image, subj_bbox, obj_bbox)
        prompt = "Answer strictly with Yes or No.\nThe red box marks the subject and the blue box marks the object.\nIs the relation true: {s} {r} {o}?"
        orig_s, orig_p = self.yes_no_score(marked, prompt.format(s=subj_name, r=relation, o=obj_name))
        swap_s, swap_p = self.yes_no_score(marked, prompt.format(s=obj_name, r=relation, o=subj_name))
        return orig_s, swap_s, {"token_probs": {"original": orig_p, "swapped": swap_p}, "marker_mode": marked.mode}

class SigLIPMixin(_TransformersBackendMixin):
    @classmethod
    def load_shared_runtime(cls, spec: BackendSpec, config: ExperimentConfig) -> Tuple[Dict[str, Any], float]:
        normalized_spec = spec
        if spec.model_path:
            normalized_spec = BackendSpec(spec.kind, resolve_siglip_model_path(spec.model_path), spec.checkpoint_path)

        import torch
        from transformers import AutoModel, AutoProcessor

        start = time.perf_counter()
        model_path = normalized_spec.model_path or _default_siglip_model_path()
        use_cuda = torch.cuda.is_available() and not config.use_cpu

        # Check if this is a SigLIP 2 model (has text encoder) or vision-only
        is_siglip2 = "siglip2" in model_path.lower()

        if is_siglip2:
            # SigLIP 2 has both image and text encoders
            proc = AutoProcessor.from_pretrained(model_path)
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if use_cuda else torch.float32,
                device_map="auto" if use_cuda else None
            )
            use_siglip2 = True
            device_map_used = use_cuda
        else:
            # Fall back to CLIP for vision-only SigLIP models
            from transformers import CLIPProcessor, CLIPModel
            proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                torch_dtype=torch.float16 if use_cuda else torch.float32,
                device_map="auto" if use_cuda else None
            )
            use_siglip2 = False
            device_map_used = use_cuda

        # Move to GPU only if device_map wasn't used
        if not device_map_used and use_cuda:
            model.to("cuda")
        model.eval()
        load_time_ms = (time.perf_counter() - start) * 1000.0
        return ({
            "proc": proc,
            "model": model,
            "use_siglip2": use_siglip2,
            "_device_map_used": device_map_used,
        }, load_time_ms)

    def __init__(self, *args, shared_runtime: Optional[Dict[str, Any]] = None):
        super().__init__(*args)
        if self.spec.model_path:
            self.spec = BackendSpec(self.spec.kind, resolve_siglip_model_path(self.spec.model_path), self.spec.checkpoint_path)
        runtime = shared_runtime
        if runtime is None:
            runtime, self.model_load_time_ms = self.load_shared_runtime(self.spec, self.config)
        self.proc = runtime["proc"]
        self.model = runtime["model"]
        self.use_siglip2 = runtime["use_siglip2"]
        self._device_map_used = runtime["_device_map_used"]

    @property
    def device(self):
        """Get the device the model is on."""
        if hasattr(self.model, "device"):
            return self.model.device
        return next(self.model.parameters()).device  # Fallback for models without device attr

    def image_text_sim(self, image: Image.Image, text: str) -> float:
        import torch

        if self.use_siglip2:
            # SigLIP 2: use separate image and text feature extraction
            inputs_img = self.proc(images=image, return_tensors="pt")
            inputs_txt = self.proc(text=[text], padding="max_length", max_length=64, return_tensors="pt")

            inputs_img = {k: v.to(self.model.device) for k, v in inputs_img.items()}
            inputs_txt = {k: v.to(self.model.device) for k, v in inputs_txt.items()}

            with torch.inference_mode():
                img_out = self.model.get_image_features(**inputs_img)
                txt_out = self.model.get_text_features(**inputs_txt)

                # Extract pooled embeddings (not last_hidden_state which gives full sequence)
                # Image features: typically [batch, hidden_dim] or [batch, patches, hidden_dim]
                # Text features: typically [batch, hidden_dim] or [batch, seq_len, hidden_dim]
                if hasattr(img_out, "image_embeds"):
                    img_features = img_out.image_embeds  # Pooled image embedding
                elif hasattr(img_out, "last_hidden_state"):
                    img_features = img_out.last_hidden_state[:, 0, :]  # Take CLS token
                else:
                    img_features = img_out
                    if img_features.dim() > 2:
                        img_features = img_features[:, 0, :]  # Take first token/pool

                if hasattr(txt_out, "text_embeds"):
                    txt_features = txt_out.text_embeds  # Pooled text embedding
                elif hasattr(txt_out, "last_hidden_state"):
                    txt_features = txt_out.last_hidden_state[:, 0, :]  # Take CLS token
                else:
                    txt_features = txt_out
                    if txt_features.dim() > 2:
                        txt_features = txt_features[:, 0, :]  # Take first token/pool

                # Ensure both are 2D tensors [batch, hidden_dim]
                img_features = img_features.flatten(0, 1) if img_features.dim() > 2 else img_features
                txt_features = txt_features.flatten(0, 1) if txt_features.dim() > 2 else txt_features

                # Compute cosine similarity
                img_features = torch.nn.functional.normalize(img_features, dim=-1)
                txt_features = torch.nn.functional.normalize(txt_features, dim=-1)
                similarity = (img_features @ txt_features.T)[0][0].item()
                return float(similarity)
        else:
            # CLIP fallback
            inputs = self.proc(text=[text], images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs)
                similarity = outputs.logits_per_image[0][0].item()
                return float(1.0 / (1.0 + torch.exp(-similarity)))  # sigmoid to 0-1 range

class SigLIPAttributeScorer(SigLIPMixin, AttributeScorerBackend):
    def _evaluate_attribute(self, crop, entity_name, attribute):
        return self.image_text_sim(crop, f"A photo of a {attribute} {entity_name}"), {}

class VLMAttributeScorer(_VisionLanguageMixin, AttributeScorerBackend):
    def _evaluate_attribute(self, crop, entity_name, attribute):
        score, probs = self.yes_no_score(crop, f"Answer strictly with Yes or No.\nDoes this crop show a {entity_name} with attribute '{attribute}'?")
        return score, {"token_probs": probs}

class SigLIPRelationScorer(SigLIPMixin, RelationScorerBackend):
    def _evaluate_relation(self, image, subj_bbox, obj_bbox, relation, subj_name, obj_name):
        crop = image.crop(tuple(union_bbox(subj_bbox, obj_bbox, self.config.stage3_margin_ratio, image.size)))
        return self.image_text_sim(crop, f"{subj_name} {relation} {obj_name}"), self.image_text_sim(crop, f"{obj_name} {relation} {subj_name}"), {"union_bbox": crop.getbbox()}

class VLMRelationScorer( _VisionLanguageMixin, RelationScorerBackend):
    def _evaluate_relation(self, image, subj_bbox, obj_bbox, relation, subj_name, obj_name):
        marked = draw_relation_markers(image, subj_bbox, obj_bbox)
        prompt = "Answer strictly with Yes or No.\nThe red box marks the subject and the blue box marks the object.\nIs the relation true: {s} {r} {o}?"
        orig_s, orig_p = self.yes_no_score(marked, prompt.format(s=subj_name, r=relation, o=obj_name))
        swap_s, swap_p = self.yes_no_score(marked, prompt.format(s=obj_name, r=relation, o=subj_name))
        return orig_s, swap_s, {"token_probs": {"original": orig_p, "swapped": swap_p}, "marker_mode": marked.mode}


def extract_scene_graph(prompt_text: str) -> Dict[str, Any]:
    text = prompt_text.split("Current Task:")[-1] if "Current Task:" in prompt_text else prompt_text
    try:
        obj_sec = text.split("Objects:", 1)[1].split("Relationships:", 1)[0]
        rel_sec = text.split("Relationships:", 1)[1].split("[Step-by-Step Reasoning]", 1)[0]
    except IndexError:
        return {"error": "Could not find expected Objects or Relationships sections."}

    objects, relations, current_obj = [], [], None
    for line in obj_sec.strip().splitlines():
        line = line.strip()
        if not line: continue
        if line.startswith("-") and "(object id" in line:
            name = line[1:].split("(object id", 1)[0].strip().split(" ", 1)[-1].strip()
            obj_id = int(line.split(":")[-1].split(")")[0].strip())
            current_obj = {"id": obj_id, "name": name, "attributes": []}
            objects.append(current_obj)
        elif current_obj and line.startswith("-"):
            current_obj["attributes"].append(line[1:].strip())

    for line in rel_sec.strip().splitlines():
        tokens = line.strip()[1:].strip().split()
        if line.strip().startswith("- Object") and len(tokens) >= 5:
            relations.append({"subject": int(tokens[1]), "relation": " ".join(tokens[2:-2]), "object": int(tokens[-1])})

    return {"objects": objects, "relations": relations}


def apply_batch_results(batch_results, all_results):
    for item_id, result in batch_results.items():
        all_results[item_id] = result


def summarize_results(results):
    pass

def _hash_val(*parts): return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16) / float(16**12 - 1)

class MockPipeline(NodeDetectorBackend, AttributeScorerBackend, RelationScorerBackend):
    def detect_nodes(self, image: Image.Image, item: ExperimentItem) -> Dict[str, Any]:
        nodes = []
        for e in item.scene_graph.get("objects", []):
            conf = 0.35 + 0.64 * _hash_val(item.image_id, self.backend_id, e.get("id"))
            box = clamp_bbox([int(image.size[0] * (0.05 + 0.4 * _hash_val(e.get("id"), "x"))), int(image.size[1] * (0.05 + 0.4 * _hash_val(e.get("id"), "y"))), image.size[0], image.size[1]], *image.size)
            nodes.append({"id": e.get("id"), "name": e.get("name"), "bbox": box if conf >= self.config.node_confidence_threshold else None, "confidence": conf, "passed": conf >= self.config.node_confidence_threshold, "score": 1.0 if conf >= self.config.node_confidence_threshold else 0.0})
        return {"backend": self.backend_id, "nodes": nodes, "fidelity_score": safe_mean(n["score"] for n in nodes)}
    def _evaluate_attribute(self, crop, entity_name, attribute): return 0.1 + 0.8 * _hash_val(entity_name, attribute, crop.size), {}
    def _evaluate_relation(self, image, s_bbox, o_bbox, relation, s_name, o_name):
        s = 0.1 + 0.8 * _hash_val(s_name, relation, o_name)
        return s, max(0.0, s - 0.25), {}

class UnavailableBackend(NodeDetectorBackend, AttributeScorerBackend, RelationScorerBackend):
    def detect_nodes(self, *a): raise NotImplementedError("Not Implemented")
    def _evaluate_attribute(self, *a): raise NotImplementedError("Not Implemented")
    def _evaluate_relation(self, *a): raise NotImplementedError("Not Implemented")

def _load_shared_runtime_for_backend(backend_id: str, spec: BackendSpec, config: ExperimentConfig) -> Tuple[Optional[Dict[str, Any]], float]:
    runtime_key = _backend_runtime_key(backend_id, spec, config)
    if runtime_key is None:
        return None, 0.0

    if runtime_key[0] == "siglip":
        return SigLIPMixin.load_shared_runtime(spec, config)
    if runtime_key[0] == "qwen-hf":
        return _VisionLanguageMixin.load_shared_runtime(spec, config)
    if runtime_key[0] == "qwen-vllm":
        return _QwenVLLMMixin.load_shared_runtime(spec, config)
    return None, 0.0

def build_backend(backend_id: str, spec: BackendSpec, config: ExperimentConfig, shared_runtimes: Optional[Dict[Tuple[Any, ...], Dict[str, Any]]] = None) -> Any:
    k = spec.kind.lower()
    if k == "mock": return MockPipeline(backend_id, spec, config)

    runtime_key = _backend_runtime_key(backend_id, spec, config)
    shared_runtime = None
    if runtime_key is not None and shared_runtimes is not None:
        cached = shared_runtimes.get(runtime_key)
        if cached is None:
            runtime, load_time_ms = _load_shared_runtime_for_backend(backend_id, spec, config)
            cached = {"runtime": runtime, "load_time_ms": load_time_ms}
            shared_runtimes[runtime_key] = cached
        shared_runtime = cached["runtime"]

    match backend_id:
        case "E1": return DinoClipNodeDetector(backend_id, spec, config) if k in {"grounding", "grounding-dino", "hf-grounding"} else UnavailableBackend(backend_id, spec, config)
        case "V1": return QwenNodeDetectorVLLM(backend_id, spec, config, shared_runtime=shared_runtime) if config.use_vllm else QwenNodeDetector(backend_id, spec, config, shared_runtime=shared_runtime) if "qwen" in k else UnavailableBackend(backend_id, spec, config)
        case "E2": return SigLIPAttributeScorer(backend_id, spec, config, shared_runtime=shared_runtime) if "siglip" in k else UnavailableBackend(backend_id, spec, config)
        case "V2": return QwenAttributeClassifierVLLM(backend_id, spec, config, shared_runtime=shared_runtime) if config.use_vllm else VLMAttributeScorer(backend_id, spec, config, shared_runtime=shared_runtime) if k in {"llava", "llava-next", "qwen", "qwen-vl"} else UnavailableBackend(backend_id, spec, config)
        case "E3": return SigLIPRelationScorer(backend_id, spec, config, shared_runtime=shared_runtime) if "siglip" in k else UnavailableBackend(backend_id, spec, config)
        case "V3": return QwenRelationScorerVLLM(backend_id, spec, config, shared_runtime=shared_runtime) if config.use_vllm else VLMRelationScorer(backend_id, spec, config, shared_runtime=shared_runtime) if "qwen" in k else UnavailableBackend(backend_id, spec, config)
    return UnavailableBackend(backend_id, spec, config)

# --- Geometry & Metric Utils ---
def safe_mean(v: Iterable) -> Optional[float]: 
    cleaned = [float(x) for x in v if x is not None]
    return sum(cleaned) / len(cleaned) if cleaned else None

def clamp_bbox(b: Sequence[float], w: int, h: int) -> List[int]: return [max(0, min(int(round(b[0])), w-1)), max(0, min(int(round(b[1])), h-1)), max(1, min(int(round(b[2])), w)), max(1, min(int(round(b[3])), h))]
def normalized_bbox_to_pixel(b: Sequence[float], w: int, h: int) -> List[int]: return clamp_bbox([w*b[0]/1000.0, h*b[1]/1000.0, w*b[2]/1000.0, h*b[3]/1000.0], w, h)
def parse_stage1_localization(raw: str, size: Tuple[int, int]) -> List[List[int]]:
    if isinstance(raw, str):
        data = json.loads(raw) if "{" in raw or "[" in raw else {}
        boxes = data.get("boxes", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    else:
        boxes = raw if isinstance(raw, list) else []
    result = []
    for b in boxes:
        if isinstance(b, dict):
            bbox = b.get("bbox", b.get("box", list(b.values())[0] if b else None))
            if bbox is not None and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                result.append(normalized_bbox_to_pixel(bbox, *size))
        elif isinstance(b, (list, tuple)) and len(b) >= 4:
            result.append(normalized_bbox_to_pixel(b, *size))
    return result
def prepare_square_crop(image: Image.Image, bbox: Sequence[int], size: int) -> Image.Image:
    c = image.crop(tuple(bbox)); s = max(c.size); cv = Image.new(image.mode, (s, s))
    cv.paste(c, ((s - c.size[0]) // 2, (s - c.size[1]) // 2)); return cv.resize((size, size))
def union_bbox(a: Sequence[int], b: Sequence[int], margin: float, size: Tuple[int, int]) -> List[int]:
    w, h = max(a[2], b[2]) - min(a[0], b[0]), max(a[3], b[3]) - min(a[1], b[1])
    return clamp_bbox([min(a[0], b[0]) - w*margin, min(a[1], b[1]) - h*margin, max(a[2], b[2]) + w*margin, max(a[3], b[3]) + h*margin], *size)
def draw_relation_markers(img: Image.Image, s_bbox: Sequence[int], o_bbox: Sequence[int]) -> Image.Image:
    arr = np.array(img.convert("RGB")); cv2.rectangle(arr, (s_bbox[0], s_bbox[1]), (s_bbox[2], s_bbox[3]), (255, 0, 0), 2); cv2.rectangle(arr, (o_bbox[0], o_bbox[1]), (o_bbox[2], o_bbox[3]), (0, 0, 255), 2); return Image.fromarray(arr)
def calibrate_score(raw: float, mode: str, scale: float, bias: float) -> float:
    s = raw * scale + bias
    return max(0.0, min(1.0, s)) if mode in ("identity", "clip") else 1.0 / (1.0 + math.exp(-s))

# --- Main Runner ---
def load_experiment_items(config: ExperimentConfig) -> List[ExperimentItem]:
    """Load experiment items by iterating through the image directory."""
    items = []

    # 1. Load the prompt data into a list for lookups
    if config.prompts_file:
        try:
            prompts_data = load_json_or_jsonl(config.prompts_file)
        except Exception as e:
            raise Exception(f"Couldn't load prompts file: {e}")
    else:
        raise ValueError("Prompts file is required to match images to data.")

    # 2. Iterate through files in the image directory
    if not os.path.exists(config.images_dir):
        raise FileNotFoundError(f"Directory not found: {config.images_dir}")

    # Filtering for .png files and sorting to maintain order
    image_files = sorted([f for f in os.listdir(config.images_dir) if f.endswith(".png")])

    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(config.images_dir, filename)
        
        try:
            # Extract the prompt index from "00XX-{gen}.png"
            # This splits by hyphen and takes the first part as the integer index
            prompt_idx = int(re.split('[-_]', filename)[0])
            
            # Ensure the index exists in our prompts data
            if prompt_idx >= len(prompts_data):
                print(f"Warning: Index {prompt_idx} from file {filename} out of range.")
                continue

            entry = prompts_data[prompt_idx]
            scene_graph = extract_scene_graph(entry["meta_prompt"]["prompt"]) if "meta_prompt" in entry else None
            prompt = entry["prompt"]
            image_id = filename.replace(".png", "")

            items.append(ExperimentItem(
                prompt_index=prompt_idx,
                image_id=str(image_id),
                prompt=str(prompt),
                image_path=str(image_path),
                scene_graph=scene_graph
            ))
        except (ValueError, IndexError) as e:
            print(f"Skipping {filename}: Could not parse index or find matching prompt.")
            continue

    # 3. Apply filtering/slicing
    if config.start_idx > 0:
        items = items[config.start_idx:]
    if config.end_idx is not None:
        items = items[:config.end_idx]
    if config.limit is not None:
        items = items[:config.limit]

    return items

# def load_experiment_items(config: ExperimentConfig) -> List[ExperimentItem]:
#     """Load experiment items from prompts file and image directory."""
#     items = []

#     if config.prompts_file:
#         try:
#             prompts_data = load_json_or_jsonl(config.prompts_file)
#         except Exception as e:
#             raise Exception("Couldn't load prompts file.")

#         for idx in tqdm(range(len(prompts_data)), desc="Loading items"):
#             entry = prompts_data[idx]
#             scene_graph = extract_scene_graph(entry["meta_prompt"]["prompt"]) if "meta_prompt" in entry else None
#             prompt = entry["prompt"]

#             for i in range(config.generation):
#                 if config.prompts_file:
#                     image_path = image_path_from_pattern(config.image_pattern, config.images_dir, idx, i + 1)
#                 else:
#                     image_path = os.path.join(config.images_dir, scene_graph["filename"])
#                 if not os.path.exists(image_path):
#                     print("Didn't find", image_path)
#                     continue

#                 image_id = os.path.basename(image_path).split(".png")[0]

#                 items.append(ExperimentItem(
#                             prompt_index=idx,
#                             image_id=str(image_id),
#                             prompt=str(prompt),
#                             image_path=str(image_path),
#                             scene_graph=scene_graph
#                         ))

#     # Apply filtering
#     if config.start_idx > 0:
#         items = items[config.start_idx:]
#     if config.end_idx is not None:
#         items = items[:config.end_idx]
#     if config.limit is not None:
#         items = items[:config.limit]

#     return items

def run_ablation_experiment(config: ExperimentConfig, items=None, backends=None) -> Dict[str, Any]:
    import torch
    wt = {"node": config.weights.node, "attribute": config.weights.attribute, "relation": config.weights.relation}
    norm_wts = {k: v / sum(wt.values()) for k, v in wt.items()}
    items = items or load_experiment_items(config)

    # Determine which backends to use based on config or defaults to all
    if hasattr(config, 'selected_backends') and config.selected_backends:
        selected_backends = config.selected_backends
    else:
        selected_backends = set(STAGE1_VARIANTS + STAGE2_VARIANTS + STAGE3_VARIANTS)

    # Filter to valid stage variants
    stage1_selected = selected_backends.intersection(STAGE1_VARIANTS) or STAGE1_VARIANTS
    stage2_selected = selected_backends.intersection(STAGE2_VARIANTS) or STAGE2_VARIANTS
    stage3_selected = selected_backends.intersection(STAGE3_VARIANTS) or STAGE3_VARIANTS

    # Only build the selected backends to save memory
    backends_to_build = stage1_selected.union(stage2_selected).union(stage3_selected)
    shared_runtimes: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    bm = backends or {b: build_backend(b, config.backend_specs[b], config, shared_runtimes=shared_runtimes) for b in backends_to_build}

    # Only create permutations for selected backends
    rows_by_perm = {f"{s1}-{s2}-{s3}": [] for s1 in stage1_selected for s2 in stage2_selected for s3 in stage3_selected}
    time_call = lambda f, *args: (lambda st=time.perf_counter(), res=f(*args): (res, (time.perf_counter() - st) * 1000.0))()

    print(f"Running with backends: stage1={stage1_selected}, stage2={stage2_selected}, stage3={stage3_selected}")
    print(f"Total permutations: {len(rows_by_perm)}")
    print(f"Total items to process: {len(items)}")

    for item in tqdm(items, desc="Processing items"):
        with Image.open(item.image_path) as img_h: img = img_h.convert("RGB")
        # Only compute for selected backends
        st1_cache = {b: dict(zip(["res", "lat"], time_call(bm[b].detect_nodes, img, item))) for b in stage1_selected}
        st2_cache = {(b1, b2): dict(zip(["res", "lat"], time_call(bm[b2].score_attributes, img, item, st1_cache[b1]["res"]))) for b1 in stage1_selected for b2 in stage2_selected}
        st3_cache = {(b1, b3): dict(zip(["res", "lat"], time_call(bm[b3].score_relations, img, item, st1_cache[b1]["res"]))) for b1 in stage1_selected for b3 in stage3_selected}

        for perm in rows_by_perm.keys():
            b1, b2, b3 = perm.split("-")
            sc = {"node": st1_cache[b1]["res"].get("fidelity_score"), "attribute": st2_cache[(b1, b2)]["res"].get("binding_score"), "relation": st3_cache[(b1, b3)]["res"].get("relation_score")}
            act_wts = {k: norm_wts[k] / sum(norm_wts[x] for x in sc if sc[x] is not None) for k in sc if sc[k] is not None} if any(v is not None for v in sc.values()) else {}

            rows_by_perm[perm].append({
                "image_id": item.image_id, "prompt": item.prompt, "permutation": perm, "final_score": sum(sc[k] * act_wts[k] for k in act_wts),
                "st1_res": st1_cache[b1]["res"], 
                "st2_res": st2_cache[(b1, b2)]["res"], 
                "st3_res": st3_cache[(b1, b3)]["res"], 
                "latency_ms": {"total": st1_cache[b1]["lat"] + st2_cache[(b1, b2)]["lat"] + st3_cache[(b1, b3)]["lat"]}
            }) # Condensed dict for brevity

            
    return {"config": asdict(config), "items_total": len(items), "permutations": rows_by_perm}

def serialize_config(config: ExperimentConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["weights"] = asdict(config.weights)
    payload["label_config"] = asdict(config.label_config)
    payload["backend_specs"] = {key: asdict(value) for key, value in config.backend_specs.items()}
    return payload

def _json_default(obj):
    """Custom JSON encoder for sets, dataclasses, and other non-serializable types."""
    if isinstance(obj, set):
        return list(obj)
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
        return asdict(obj)
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return str(obj)  # Fallback to string representation

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames: fieldnames.append(key)
            
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def write_experiment_outputs(report: Mapping[str, Any], output_dir: str) -> Dict[str, str]:
    root = Path(output_dir)
    paths = {
        "run_metadata": root / "run_metadata.json",
        "aggregate_json": root / "aggregate_matrix.json",
        "aggregate_csv": root / "aggregate_matrix.csv",
        "latency_json": root / "latency_report.json",
        "relation_json": root / "relation_swap_report.json",
    }
    
    write_json(paths["run_metadata"], {"config": report["config"], "items_total": report["items_total"]})
    write_json(paths["aggregate_json"], report.get("aggregate_matrix", []))
    write_csv(paths["aggregate_csv"], report.get("aggregate_matrix", []))
    write_json(paths["latency_json"], report.get("latency_report", {}))
    write_json(paths["relation_json"], report.get("relation_swap_report", {}))

    if report.get("correlation_report"):
        paths["correlation_json"] = root / "correlation_report.json"
        write_json(paths["correlation_json"], report["correlation_report"])

    for perm, payload in report.get("permutations", {}).items():
        write_json(root / "permutations" / f"{perm}_details.json", payload)

    return {k: str(v) for k, v in paths.items()}

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run 8-way graph-grounded alignment ablations.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prompts-file", default=None)
    p.add_argument("--sg-file", default=None)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--image-pattern", default="{index:04d}-{generation}.png")
    p.add_argument("--generation", type=int, default=1)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--human-score-file", default=None)
    p.add_argument("--label-key-field", default="image_id")
    p.add_argument("--label-score-field", default="score")
    p.add_argument("--result-key-field", default="image_id")
    p.add_argument("--weight-node", type=float, default=0.3)
    p.add_argument("--weight-attribute", type=float, default=0.3)
    p.add_argument("--weight-relation", type=float, default=0.3)
    p.add_argument("--node-confidence-threshold", type=float, default=0.5)
    p.add_argument("--node-nms-threshold", type=float, default=0.3)
    p.add_argument("--stage2-crop-size", type=int, default=384)
    p.add_argument("--stage2-calibration", default="clip", choices=["identity", "clip", "sigmoid"])
    p.add_argument("--stage2-calibration-scale", type=float, default=1.0)
    p.add_argument("--stage2-calibration-bias", type=float, default=0.0)
    p.add_argument("--stage3-margin-ratio", type=float, default=0.1)
    p.add_argument("--include-model-load-time", action="store_true")
    p.add_argument("--backends", default=None, help="Comma-separated list of backends to use (e.g., 'E1,E2,E3' for encoder-only pipeline). Defaults to all backends.")
    p.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU (slower but uses less memory)")
    p.add_argument("--low-vram", action="store_true", help="Use lower VRAM settings (models loaded on CPU, moved to GPU only during inference)")
    p.add_argument("--use-vllm", action="store_true", help="Use a vLLM OpenAI-compatible server for Qwen inference")
    p.add_argument("--vllm-api-base", default=os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8000/v1"), help="Base URL for the vLLM OpenAI-compatible API")
    p.add_argument("--vllm-api-key", default=os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY"), help="Optional API key for the vLLM server")
    p.add_argument("--vllm-temperature", type=float, default=None, help="Optional temperature to send to the vLLM server")
    p.add_argument("--vllm-max-tokens", type=int, default=None, help="Optional max_tokens for general vLLM requests")
    p.add_argument("--vllm-yes-no-max-tokens", type=int, default=None, help="Optional max_tokens for vLLM yes/no requests")
    p.add_argument("--max-text-length", type=int, default=64, help="Max text length for SigLIP 2 models (default: 64)")
    p.add_argument("--torch-cuda-mem-frac", type=float, default=0.8, help="Fraction of GPU memory to use (for device_map='auto')")

    # Condensed repetitive backend argument parsing
    for b in ("e1", "v1", "e2", "v2", "e3", "v3"):
        p.add_argument(f"--{b}-backend-kind", default=None) # was "mock")

    for m in ("eupe", "qwen", "siglip", "llava"):
        p.add_argument(f"--{m}-model-path", default=None)
        p.add_argument(f"--{m}-checkpoint-path", default=None)

    return p
def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    none_vals = {None, "None"}
    return ExperimentConfig(
        output_dir=args.output_dir,
        prompts_file=None if args.prompts_file in none_vals else args.prompts_file,
        sg_file=None if args.sg_file in none_vals else args.sg_file,
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
            path=None if args.human_score_file in none_vals else args.human_score_file,
            key_field=args.label_key_field,
            score_field=args.label_score_field,
            result_key_field=args.result_key_field,
        ),
        backend_specs={
            "E1": BackendSpec(args.e1_backend_kind, args.eupe_model_path, args.eupe_checkpoint_path),
            "V1": BackendSpec(args.v1_backend_kind, args.qwen_model_path, args.qwen_checkpoint_path),
            "E2": BackendSpec(args.e2_backend_kind, args.siglip_model_path, args.siglip_checkpoint_path),
            "V2": BackendSpec(args.v2_backend_kind, args.qwen_model_path, args.qwen_checkpoint_path),
            "E3": BackendSpec(args.e3_backend_kind, args.siglip_model_path, args.siglip_checkpoint_path),
            "V3": BackendSpec(args.v3_backend_kind, args.qwen_model_path, args.qwen_checkpoint_path),
        },
        selected_backends=set(args.backends.split(",")) if args.backends else None,
        use_cpu=args.cpu,
        low_vram=args.low_vram,
        use_vllm=args.use_vllm,
        max_text_length=args.max_text_length,
        torch_cuda_mem_frac=args.torch_cuda_mem_frac,
        vllm_api_base=args.vllm_api_base,
        vllm_api_key=args.vllm_api_key,
        vllm_temperature=args.vllm_temperature,
        vllm_max_tokens=args.vllm_max_tokens,
        vllm_yes_no_max_tokens=args.vllm_yes_no_max_tokens,
    )

if __name__ == "__main__":
    import sys
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    print(f"Running ablation experiment with config:")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Prompts file: {config.prompts_file}")
    print(f"  Images dir: {config.images_dir}")
    print(f"  Backend specs: {config.backend_specs}")
    print(f"  Use vLLM: {config.use_vllm}")
    if config.use_vllm:
        print(f"  vLLM API base: {config.vllm_api_base}")
    report = run_ablation_experiment(config)
    write_experiment_outputs(report, config.output_dir)
