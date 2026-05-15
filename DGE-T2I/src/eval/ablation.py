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

try:
    from .modules.processing import (
        apply_batch_results,
        build_attribute_prompt,
        build_object_prompt,
        build_relation_prompt,
        extract_json,
        image_path_from_pattern,
        load_json_or_jsonl,
        normalize_answer,
        normalize_bbox,
        normalize_visible,
        summarize_results,
        extract_scene_graph,
    )
except ImportError:
    from modules.processing import (
        apply_batch_results,
        build_attribute_prompt,
        build_object_prompt,
        build_relation_prompt,
        extract_json,
        image_path_from_pattern,
        load_json_or_jsonl,
        normalize_answer,
        normalize_bbox,
        normalize_visible,
        summarize_results,
        extract_scene_graph,
    )

from tqdm import tqdm

STAGE1_VARIANTS = ("E1", "V1")
STAGE2_VARIANTS = ("E2", "V2", "S2")
DEFAULT_STAGE2_VARIANTS = ("E2", "V2")
STAGE3_VARIANTS = ("E3", "V3")

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
    start_idx: int; end_idx: Optional[int]; limit: Optional[int]; skip_indices: Tuple[int, ...]; weights: StageWeights
    node_confidence_threshold: float; node_nms_threshold: float; stage2_crop_size: int
    stage2_calibration: str; stage2_calibration_scale: float; stage2_calibration_bias: float; stage3_margin_ratio: float
    include_model_load_time: bool; label_config: LabelConfig; backend_specs: Dict[str, BackendSpec]
    selected_backends: Optional[set] = None; use_cpu: bool = False; low_vram: bool = False; use_vllm: bool = False
    max_text_length: int = 64; torch_cuda_mem_frac: float = 0.8
    vllm_api_base: str = "http://127.0.0.1:8000/v1"; vllm_api_key: Optional[str] = None
    vllm_temperature: Optional[float] = None; vllm_max_tokens: Optional[int] = None; vllm_yes_no_max_tokens: Optional[int] = None
    molmopoint_model_path: Optional[str] = None; molmopoint_checkpoint_path: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, 'selected_backends', self.selected_backends or None)

class StageBackend(ABC):
    def __init__(self, backend_id: str, spec: BackendSpec, config: ExperimentConfig):
        self.backend_id, self.spec, self.config, self.model_load_time_ms = backend_id, spec, config, 0.0

def _default_qwen_model_path() -> str:
    return "../../Qwen3-VL-8B-Instruct"

def _default_siglip_model_path() -> str:
    return "google/siglip2-so400m-patch14-384"

def _default_eva_clip_model_path() -> str:
    return "EVA02-CLIP-L-14-336"

def _default_eva_clip_checkpoint_path() -> str:
    return os.environ.get("EVA_CLIP_CHECKPOINT_PATH") or "https://huggingface.co/QuanSun/EVA-CLIP/blob/main/EVA02_CLIP_L_psz14_224to336.pt"

def _default_blip2_model_path() -> str:
    return os.environ.get("BLIP2_MODEL_PATH") or "Salesforce/blip2-itm-vit-g"

def _default_molmopoint_model_path() -> str:
    return os.environ.get("MOLMOPOINT_MODEL_PATH") or "allenai/MolmoPoint-8B"

def _default_reitr_model_path() -> str:
    return os.environ.get("REITR_CODE_DIR") or os.environ.get("RELTR_CODE_DIR") or "../../RelTR"

def _default_reitr_checkpoint_path() -> Optional[str]:
    return os.environ.get("REITR_CHECKPOINT_PATH") or os.environ.get("RELTR_CHECKPOINT_PATH")

def _default_relation_text_embedding_model_path() -> str:
    return os.environ.get("QWEN3_EMBEDDING_MODEL_PATH") or "Qwen/Qwen3-Embedding-0.6B"

def _backend_runtime_key(backend_id: str, spec: BackendSpec, config: ExperimentConfig) -> Optional[Tuple[Any, ...]]:
    kind = (spec.kind or "").lower()

    if backend_id in {"E2", "E3"} and "siglip" in kind:
        model_path = resolve_siglip_model_path(spec.model_path) if spec.model_path else _default_siglip_model_path()
        return ("siglip", model_path, spec.checkpoint_path, config.use_cpu)

    if backend_id in {"E2", "E3"} and kind in {"eva", "eva-clip", "evaclip"}:
        model_path = spec.model_path or _default_eva_clip_model_path()
        checkpoint_path = spec.checkpoint_path or _default_eva_clip_checkpoint_path()
        return ("eva-clip", model_path, checkpoint_path, config.use_cpu)

    if backend_id == "E2" and kind in {"blip2", "blip-2"}:
        model_path = spec.model_path or _default_blip2_model_path()
        return ("blip2", model_path, spec.checkpoint_path, config.use_cpu)

    if backend_id == "E3" and kind in {"reitr", "reltr"}:
        model_path = spec.model_path or _default_reitr_model_path()
        checkpoint_path = spec.checkpoint_path or _default_reitr_checkpoint_path()
        return ("reitr", model_path, checkpoint_path, config.use_cpu)

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
        from scipy.optimize import linear_sum_assignment
        labels = [e.get("name", "") for e in item.scene_graph.get("objects", [])]
        print("finding labels, ",labels)
        dino_text = " . ".join(labels) + " ."
        dino_inputs = self.dino_proc(images=image, text=dino_text, return_tensors="pt").to(self.device)
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

        # 3. Compute CLIP features separately to get a proper (num_labels x num_crops) matrix
        clip_texts = [f"a photo of a {label}" for label in labels]

        text_inputs = self.clip_proc(
            text=clip_texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        image_inputs = self.clip_proc(
            images=crops, return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            text_features = self.clip_model.get_text_features(**text_inputs)    # (num_labels, D)
            image_features = self.clip_model.get_image_features(**image_inputs) # (num_crops, D)

        # L2-normalize for cosine similarity
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # similarity[i, j] = cosine similarity between label i and crop j
        similarity = (text_features @ image_features.T).cpu().float().numpy()  # (num_labels, num_crops)

        # 4. Hungarian matching: unique 1-to-1 assignment of labels to crops
        #    Pad with dummy columns if there are fewer crops than labels so every
        #    label gets a slot (dummy slots will have similarity = -1).
        num_labels, num_crops = similarity.shape
        if num_crops < num_labels:
            padding = -1.0 * torch.ones(num_labels, num_labels - num_crops).numpy()
            import numpy as np
            similarity = np.concatenate([similarity, padding], axis=1)

        row_ind, col_ind = linear_sum_assignment(-similarity)  # maximise similarity

        # 5. Build node results
        for i, entity in enumerate(item.scene_graph.get("objects", [])):
            assigned_col = col_ind[row_ind == i]

            # Label got a dummy column (no real crop available)
            if len(assigned_col) == 0 or assigned_col[0] >= num_crops:
                nodes.append({
                    "id": entity.get("id"),
                    "name": entity.get("name"),
                    "bbox": None,
                    "confidence": 0.0,
                    "passed": False,
                    "score": 0.0,
                })
                continue

            j = int(assigned_col[0])
            confidence = float(similarity[i, j])
            best_box = valid_boxes[j]
            passed = confidence >= self.config.node_confidence_threshold

            nodes.append({
                "id": entity.get("id"),
                "name": entity.get("name"),
                "bbox": best_box if passed else None,
                "confidence": confidence,
                "passed": passed,
                "score": 1.0 if passed else 0.0,
            })

        return {
            "backend": self.backend_id,
            "nodes": nodes,
            "fidelity_score": safe_mean(n["score"] for n in nodes),
        }
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
    """Qwen node detector using the same per-object prompts as the eval pipeline."""

    def detect_nodes(self, image: Image.Image, item: ExperimentItem) -> Dict[str, Any]:
        objects = item.scene_graph.get("objects", [])
        if not objects:
            return {"backend": self.backend_id, "nodes": [], "fidelity_score": 1.0}

        nodes = []
        width, height = image.size
        for entity in objects:
            raw = self.generate_text(image, build_object_prompt(width, height, entity))
            nodes.append(_node_from_eval_object_response(raw, entity, image.size, self.config.node_confidence_threshold))

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
    """Qwen node detector using vLLM and the eval pipeline's per-object prompts."""

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
        objects = item.scene_graph.get("objects", [])
        if not objects:
            return {"backend": self.backend_id, "nodes": [], "fidelity_score": 1.0}

        width, height = image.size
        prompts = [build_object_prompt(width, height, entity) for entity in objects]
        raw_outputs = self._chat_texts([self._messages_for_image(image, prompt) for prompt in prompts], self.sampling_params)
        nodes = [
            _node_from_eval_object_response(raw, entity, image.size, self.config.node_confidence_threshold)
            for raw, entity in zip(raw_outputs, objects)
        ]

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
        results = []
        requests = []
        request_indices = []
        width, height = image.size

        for entity in item.scene_graph.get("objects", []):
            for attr in entity.get("attributes", []):
                prompt = build_attribute_prompt(width, height, entity, attr)
                request_indices.append(len(results))
                requests.append((image, prompt))
                results.append({"id": entity.get("id"), "name": entity.get("name"), "attribute": attr, "skipped": False})

        raw_outputs = self._chat_texts([self._messages_for_image(img, prompt) for img, prompt in requests], self.sampling_params)
        for result_idx, raw in zip(request_indices, raw_outputs):
            raw_score, answer = _score_eval_yes_no_json(raw)
            cal_score = calibrate_score(raw_score, self.config.stage2_calibration, self.config.stage2_calibration_scale, self.config.stage2_calibration_bias)
            results[result_idx].update({"score": raw_score, "calibrated_score": cal_score, "answer": answer})

        return {"backend": self.backend_id, "crop_size": self.config.stage2_crop_size, "attributes": results, "binding_score": safe_mean(e.get("calibrated_score") for e in results if not e.get("skipped"))}

    def _evaluate_attribute(self, crop, entity_name, attribute):
        raise NotImplementedError("QwenAttributeClassifierVLLM uses score_attributes with eval-pipeline prompts.")

class _MolmoPointCountAttributeMixin:
    def _init_molmopoint_count_scorer(self) -> None:
        self.molmopoint_count_scorer = None

    def _get_molmopoint_count_scorer(self) -> "MolmoPointAttributeScorer":
        if self.molmopoint_count_scorer is None:
            self.molmopoint_count_scorer = MolmoPointAttributeScorer(
                f"{self.backend_id}_molmopoint",
                BackendSpec(
                    "molmopoint",
                    self.config.molmopoint_model_path or _default_molmopoint_model_path(),
                    self.config.molmopoint_checkpoint_path,
                ),
                self.config,
            )
        return self.molmopoint_count_scorer

    def _score_count_attribute_with_molmopoint(
        self,
        image: Image.Image,
        entity_name: str,
        expected_count: int,
        point_cache: Dict[str, Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        point_result = point_cache.get(entity_name)
        if point_result is None:
            point_result = self._get_molmopoint_count_scorer().point_to_objects(image, entity_name)
            point_cache[entity_name] = point_result
        observed_count = len(point_result["points"])
        return (1.0 if observed_count == expected_count else 0.0), {
            "answer": "yes" if observed_count == expected_count else "no",
            "expected_count": expected_count,
            "observed_count": observed_count,
            "points": point_result["points"],
            "point_prompt": point_result["prompt"],
            "raw_output": point_result["raw_output"],
            "count_evaluator": "molmopoint",
        }

class QwenMolmoPointAttributeClassifierVLLM(_MolmoPointCountAttributeMixin, QwenAttributeClassifierVLLM):
    def __init__(self, *args, shared_runtime: Optional[Dict[str, Any]] = None):
        super().__init__(*args, shared_runtime=shared_runtime)
        self._init_molmopoint_count_scorer()

    def score_attributes(self, image: Image.Image, item: ExperimentItem, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        results = []
        requests = []
        request_indices = []
        point_cache: Dict[str, Dict[str, Any]] = {}
        width, height = image.size

        for entity in item.scene_graph.get("objects", []):
            entity_name = str(entity.get("name", "")).strip()
            for attr in entity.get("attributes", []):
                expected_count = parse_count_attribute(attr)
                result = {"id": entity.get("id"), "name": entity.get("name"), "attribute": attr, "skipped": False}
                if expected_count is not None:
                    if not entity_name:
                        result.update({"skipped": True, "skip_reason": "missing_entity_name"})
                    else:
                        raw_score, extra = self._score_count_attribute_with_molmopoint(image, entity_name, expected_count, point_cache)
                        cal_score = calibrate_score(raw_score, self.config.stage2_calibration, self.config.stage2_calibration_scale, self.config.stage2_calibration_bias)
                        result.update({"score": raw_score, "calibrated_score": cal_score, **extra})
                    results.append(result)
                    continue

                prompt = build_attribute_prompt(width, height, entity, attr)
                request_indices.append(len(results))
                requests.append((image, prompt))
                results.append(result)

        raw_outputs = self._chat_texts([self._messages_for_image(img, prompt) for img, prompt in requests], self.sampling_params)
        for result_idx, raw in zip(request_indices, raw_outputs):
            raw_score, answer = _score_eval_yes_no_json(raw)
            cal_score = calibrate_score(raw_score, self.config.stage2_calibration, self.config.stage2_calibration_scale, self.config.stage2_calibration_bias)
            results[result_idx].update({"score": raw_score, "calibrated_score": cal_score, "answer": answer, "count_evaluator": "qwen"})

        return {"backend": self.backend_id, "crop_size": None, "attributes": results, "binding_score": safe_mean(e.get("calibrated_score") for e in results if not e.get("skipped"))}

class QwenRelationScorerVLLM(QwenNodeDetectorVLLM, RelationScorerBackend):
    def score_relations(self, image: Image.Image, item: ExperimentItem, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        entity_map = {e["id"]: e for e in item.scene_graph.get("objects", [])}
        results = []
        requests = []
        request_indices = []
        width, height = image.size

        for rel in item.scene_graph.get("relations", []):
            swapped_rel = {"subject": rel.get("object"), "relation": rel.get("relation"), "object": rel.get("subject")}
            request_indices.append(len(results))
            requests.append((image, build_relation_prompt(width, height, rel, entity_map)))
            requests.append((image, build_relation_prompt(width, height, swapped_rel, entity_map)))
            results.append({"subject": rel["subject"], "relation": rel["relation"], "object": rel["object"], "skipped": False})

        raw_outputs = self._chat_texts([self._messages_for_image(img, prompt) for img, prompt in requests], self.sampling_params)
        for result_idx, pair_start in zip(request_indices, range(0, len(raw_outputs), 2)):
            orig_raw, orig_answer = _score_eval_yes_no_json(raw_outputs[pair_start])
            swap_raw, swap_answer = _score_eval_yes_no_json(raw_outputs[pair_start + 1])
            cal = lambda s: calibrate_score(s, self.config.stage2_calibration, self.config.stage2_calibration_scale, self.config.stage2_calibration_bias)
            orig_score, swap_score = cal(orig_raw), cal(swap_raw)
            results[result_idx].update({
                "original_score": orig_score,
                "swapped_score": swap_score,
                "delta": orig_score - swap_score,
                "swap_correct": orig_score > swap_score,
                "answers": {"original": orig_answer, "swapped": swap_answer},
            })

        return {"backend": self.backend_id, "relations": results, "relation_score": safe_mean(e.get("original_score") for e in results if not e.get("skipped")), "swap_accuracy": safe_mean(1.0 if e.get("swap_correct") else 0.0 for e in results if e.get("swap_correct") is not None)}

    def _evaluate_relation(self, image, subj_bbox, obj_bbox, relation, subj_name, obj_name):
        raise NotImplementedError("QwenRelationScorerVLLM uses score_relations with eval-pipeline prompts.")

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

        image = image.convert("RGB")
        if min(image.size) < 2:
            image = image.resize((max(2, image.size[0]), max(2, image.size[1])))

        if self.use_siglip2:
            # SigLIP 2: use separate image and text feature extraction
            from transformers.image_utils import ChannelDimension
            inputs_img = self.proc(images=np.asarray(image), return_tensors="pt", input_data_format=ChannelDimension.LAST)
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
        return score_contrastive_attribute(self.image_text_sim, crop, entity_name, attribute)

class EVAClipMixin(_TransformersBackendMixin):
    @classmethod
    def load_shared_runtime(cls, spec: BackendSpec, config: ExperimentConfig) -> Tuple[Dict[str, Any], float]:
        import sys
        import torch

        start = time.perf_counter()
        model_path = spec.model_path or _default_eva_clip_model_path()
        checkpoint_path = _resolve_checkpoint_path(spec.checkpoint_path or _default_eva_clip_checkpoint_path())
        use_cuda = torch.cuda.is_available() and not config.use_cpu
        device = "cuda" if use_cuda else "cpu"

        eva_code_dir = os.environ.get("EVA_CLIP_CODE_DIR")
        if eva_code_dir and eva_code_dir not in sys.path:
            sys.path.insert(0, eva_code_dir)
        try:
            from eva_clip import create_model_and_transforms, get_tokenizer
        except ImportError as exc:
            raise ImportError(
                "EVA-CLIP backend requires the BAAI EVA-CLIP code on PYTHONPATH. "
                "Set EVA_CLIP_CODE_DIR to the EVA-CLIP/rei directory, or install the eva_clip package."
            ) from exc

        precision = "fp16" if use_cuda else "fp32"
        model, _, preprocess = create_model_and_transforms(
            model_path,
            checkpoint_path,
            precision=precision,
            device=device,
        )
        tokenizer = get_tokenizer(model_path)
        model.eval()
        load_time_ms = (time.perf_counter() - start) * 1000.0
        return ({
            "preprocess": preprocess,
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
            "use_cuda": use_cuda,
        }, load_time_ms)

    def __init__(self, *args, shared_runtime: Optional[Dict[str, Any]] = None):
        super().__init__(*args)
        runtime = shared_runtime
        if runtime is None:
            runtime, self.model_load_time_ms = self.load_shared_runtime(self.spec, self.config)
        self.preprocess = runtime["preprocess"]
        self.tokenizer = runtime["tokenizer"]
        self.model = runtime["model"]
        self.device = runtime["device"]
        self.use_cuda = runtime["use_cuda"]

    def image_text_sim(self, image: Image.Image, text: str) -> float:
        import torch

        image = image.convert("RGB")
        input_pixels = self.preprocess(image).unsqueeze(0).to(self.device)
        input_ids = self.tokenizer([text]).to(self.device)
        autocast_enabled = bool(self.use_cuda)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=autocast_enabled):
            image_features = self.model.encode_image(input_pixels)
            text_features = self.model.encode_text(input_ids)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
            return float((image_features @ text_features.T)[0][0].item())

class EVAClipAttributeScorer(EVAClipMixin, AttributeScorerBackend):
    def _evaluate_attribute(self, crop, entity_name, attribute):
        return score_contrastive_attribute(self.image_text_sim, crop, entity_name, attribute)

class BLIP2Mixin(_TransformersBackendMixin):
    @classmethod
    def load_shared_runtime(cls, spec: BackendSpec, config: ExperimentConfig) -> Tuple[Dict[str, Any], float]:
        import torch
        from transformers import AutoProcessor, Blip2ForImageTextRetrieval

        start = time.perf_counter()
        model_path = spec.model_path or _default_blip2_model_path()
        use_cuda = torch.cuda.is_available() and not config.use_cpu
        dtype = torch.float16 if use_cuda else torch.float32
        proc = AutoProcessor.from_pretrained(model_path)
        kwargs = {"torch_dtype": dtype}
        if use_cuda:
            kwargs["device_map"] = "auto"
        model = Blip2ForImageTextRetrieval.from_pretrained(model_path, **kwargs)
        if spec.checkpoint_path:
            ckpt = torch.load(spec.checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict({k.replace("module.", "", 1): v for k, v in state_dict.items()}, strict=False)
        if not use_cuda:
            model.to("cpu")
        model.eval()
        load_time_ms = (time.perf_counter() - start) * 1000.0
        return ({
            "proc": proc,
            "model": model,
            "dtype": dtype,
        }, load_time_ms)

    def __init__(self, *args, shared_runtime: Optional[Dict[str, Any]] = None):
        super().__init__(*args)
        runtime = shared_runtime
        if runtime is None:
            runtime, self.model_load_time_ms = self.load_shared_runtime(self.spec, self.config)
        self.proc = runtime["proc"]
        self.model = runtime["model"]
        self.dtype = runtime["dtype"]

    @property
    def device(self):
        if hasattr(self.model, "device"):
            return self.model.device
        return next(self.model.parameters()).device

    def _to_blip2_device(self, inputs: Any) -> Dict[str, Any]:
        import torch

        moved = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                moved[key] = value
            elif torch.is_floating_point(value):
                moved[key] = value.to(device=self.device, dtype=self.dtype)
            else:
                moved[key] = value.to(self.device)
        return moved

    def image_text_sim(self, image: Image.Image, text: str) -> float:
        import torch

        image = image.convert("RGB")
        if min(image.size) < 2:
            image = image.resize((max(2, image.size[0]), max(2, image.size[1])))

        inputs = self.proc(images=image, text=text, return_tensors="pt")
        inputs = self._to_blip2_device(inputs)
        with torch.inference_mode():
            outputs = self.model(**inputs, use_image_text_matching_head=False)
            if outputs.logits_per_image is not None:
                return float(outputs.logits_per_image[0][0].item())
            image_embeds = torch.nn.functional.normalize(outputs.image_embeds, dim=-1)
            text_embeds = torch.nn.functional.normalize(outputs.text_embeds, dim=-1)
            similarities = image_embeds @ text_embeds.T
            return float(similarities.max().item())

class BLIP2AttributeScorer(BLIP2Mixin, AttributeScorerBackend):
    def _evaluate_attribute(self, crop, entity_name, attribute):
        return score_contrastive_attribute(self.image_text_sim, crop, entity_name, attribute)

class MolmoPointAttributeScorer(AttributeScorerBackend):
    @classmethod
    def load_shared_runtime(cls, spec: BackendSpec, config: ExperimentConfig) -> Tuple[Dict[str, Any], float]:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        start = time.perf_counter()
        model_path = spec.model_path or _default_molmopoint_model_path()
        use_cuda = torch.cuda.is_available() and not config.use_cpu
        proc = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        kwargs = {"trust_remote_code": True, "dtype": "auto"}
        if use_cuda:
            kwargs["device_map"] = "auto"
        model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
        if spec.checkpoint_path:
            ckpt = torch.load(spec.checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict({k.replace("module.", "", 1): v for k, v in state_dict.items()}, strict=False)
        if not use_cuda:
            model.to("cpu")
        model.eval()
        load_time_ms = (time.perf_counter() - start) * 1000.0
        return {"proc": proc, "model": model, "use_cuda": use_cuda}, load_time_ms

    def __init__(self, *args, shared_runtime: Optional[Dict[str, Any]] = None):
        super().__init__(*args)
        runtime = shared_runtime
        if runtime is None:
            runtime, self.model_load_time_ms = self.load_shared_runtime(self.spec, self.config)
        self.proc = runtime["proc"]
        self.model = runtime["model"]
        self.use_cuda = runtime["use_cuda"]

    @property
    def device(self):
        if hasattr(self.model, "device"):
            return self.model.device
        return next(self.model.parameters()).device

    def score_attributes(self, image: Image.Image, item: ExperimentItem, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        point_cache: Dict[str, Dict[str, Any]] = {}
        results = []
        for entity in item.scene_graph.get("objects", []):
            entity_name = str(entity.get("name", "")).strip()
            for attr in entity.get("attributes", []):
                expected_count = parse_count_attribute(attr)
                if expected_count is None:
                    results.append({
                        "id": entity.get("id"),
                        "name": entity.get("name"),
                        "attribute": attr,
                        "skipped": True,
                        "skip_reason": "non_count_attribute",
                    })
                    continue
                if not entity_name:
                    results.append({
                        "id": entity.get("id"),
                        "name": entity.get("name"),
                        "attribute": attr,
                        "skipped": True,
                        "skip_reason": "missing_entity_name",
                    })
                    continue

                point_result = point_cache.get(entity_name)
                if point_result is None:
                    point_result = self.point_to_objects(image, entity_name)
                    point_cache[entity_name] = point_result
                observed_count = len(point_result["points"])
                raw_score = 1.0 if observed_count == expected_count else 0.0
                cal_score = calibrate_score(raw_score, self.config.stage2_calibration, self.config.stage2_calibration_scale, self.config.stage2_calibration_bias)
                results.append({
                    "id": entity.get("id"),
                    "name": entity.get("name"),
                    "attribute": attr,
                    "score": raw_score,
                    "calibrated_score": cal_score,
                    "skipped": False,
                    "expected_count": expected_count,
                    "observed_count": observed_count,
                    "points": point_result["points"],
                    "point_prompt": point_result["prompt"],
                    "raw_output": point_result["raw_output"],
                })

        return {
            "backend": self.backend_id,
            "crop_size": None,
            "attributes": results,
            "binding_score": safe_mean(e.get("calibrated_score") for e in results if not e.get("skipped")),
            "count_only": True,
        }

    def point_to_objects(self, image: Image.Image, entity_name: str) -> Dict[str, Any]:
        import torch

        prompt = f"Point to every {entity_name} in the image."
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image.convert("RGB")},
            ],
        }]
        inputs = self.proc.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            return_pointing_metadata=True,
        )
        metadata = inputs.pop("metadata", None)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        gen_kwargs = {"max_new_tokens": 200}
        if hasattr(self.model, "build_logit_processor_from_inputs"):
            gen_kwargs["logits_processor"] = self.model.build_logit_processor_from_inputs(inputs)

        autocast_device = "cuda" if self.use_cuda else "cpu"
        with torch.inference_mode(), torch.autocast(autocast_device, dtype=torch.bfloat16, enabled=self.use_cuda):
            output = self.model.generate(**inputs, **gen_kwargs)
        generated_tokens = output[:, inputs["input_ids"].size(1):]
        raw_output = self.proc.post_process_image_text_to_text(
            generated_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        points = self._extract_points(raw_output, metadata)
        return {"prompt": prompt, "raw_output": raw_output, "points": points}

    def _extract_points(self, raw_output: str, metadata: Optional[Dict[str, Any]]) -> List[Dict[str, float]]:
        if not metadata:
            raise RuntimeError("MolmoPoint-8B requires return_pointing_metadata=True to decode point tokens.")
        if not hasattr(self.model, "extract_image_points"):
            raise RuntimeError("MolmoPoint-8B model is missing extract_image_points; check trust_remote_code/model_path.")
        extracted = self.model.extract_image_points(
            raw_output,
            metadata["token_pooling"],
            metadata["subpatch_mapping"],
            metadata["image_sizes"],
        )
        return [
            {"object_id": int(p[0]), "image_num": int(p[1]), "x": float(p[2]), "y": float(p[3])}
            for p in extracted
        ]

    def _evaluate_attribute(self, crop, entity_name, attribute):
        raise NotImplementedError("MolmoPointAttributeScorer uses score_attributes with whole-image pointing.")

class VLMAttributeScorer(_VisionLanguageMixin, AttributeScorerBackend):
    def score_attributes(self, image: Image.Image, item: ExperimentItem, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        results = []
        width, height = image.size
        for entity in item.scene_graph.get("objects", []):
            for attr in entity.get("attributes", []):
                raw = self.generate_text(image, build_attribute_prompt(width, height, entity, attr))
                raw_score, answer = _score_eval_yes_no_json(raw)
                cal_score = calibrate_score(raw_score, self.config.stage2_calibration, self.config.stage2_calibration_scale, self.config.stage2_calibration_bias)
                results.append({
                    "id": entity.get("id"),
                    "name": entity.get("name"),
                    "attribute": attr,
                    "score": raw_score,
                    "calibrated_score": cal_score,
                    "answer": answer,
                    "skipped": False,
                })
        return {"backend": self.backend_id, "crop_size": None, "attributes": results, "binding_score": safe_mean(e.get("calibrated_score") for e in results if not e.get("skipped"))}

    def _evaluate_attribute(self, crop, entity_name, attribute):
        raise NotImplementedError("VLMAttributeScorer uses score_attributes with eval-pipeline prompts.")

class VLMMolmoPointAttributeScorer(_MolmoPointCountAttributeMixin, VLMAttributeScorer):
    def __init__(self, *args, shared_runtime: Optional[Dict[str, Any]] = None):
        super().__init__(*args, shared_runtime=shared_runtime)
        self._init_molmopoint_count_scorer()

    def score_attributes(self, image: Image.Image, item: ExperimentItem, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        results = []
        point_cache: Dict[str, Dict[str, Any]] = {}
        width, height = image.size
        for entity in item.scene_graph.get("objects", []):
            entity_name = str(entity.get("name", "")).strip()
            for attr in entity.get("attributes", []):
                expected_count = parse_count_attribute(attr)
                if expected_count is not None:
                    result = {"id": entity.get("id"), "name": entity.get("name"), "attribute": attr, "skipped": False}
                    if not entity_name:
                        result.update({"skipped": True, "skip_reason": "missing_entity_name"})
                    else:
                        raw_score, extra = self._score_count_attribute_with_molmopoint(image, entity_name, expected_count, point_cache)
                        cal_score = calibrate_score(raw_score, self.config.stage2_calibration, self.config.stage2_calibration_scale, self.config.stage2_calibration_bias)
                        result.update({"score": raw_score, "calibrated_score": cal_score, **extra})
                    results.append(result)
                    continue

                raw = self.generate_text(image, build_attribute_prompt(width, height, entity, attr))
                raw_score, answer = _score_eval_yes_no_json(raw)
                cal_score = calibrate_score(raw_score, self.config.stage2_calibration, self.config.stage2_calibration_scale, self.config.stage2_calibration_bias)
                results.append({
                    "id": entity.get("id"),
                    "name": entity.get("name"),
                    "attribute": attr,
                    "score": raw_score,
                    "calibrated_score": cal_score,
                    "answer": answer,
                    "count_evaluator": "qwen",
                    "skipped": False,
                })
        return {"backend": self.backend_id, "crop_size": None, "attributes": results, "binding_score": safe_mean(e.get("calibrated_score") for e in results if not e.get("skipped"))}

class SigLIPRelationScorer(SigLIPMixin, RelationScorerBackend):
    def _evaluate_relation(self, image, subj_bbox, obj_bbox, relation, subj_name, obj_name):
        crop = image.crop(tuple(union_bbox(subj_bbox, obj_bbox, self.config.stage3_margin_ratio, image.size)))
        return self.image_text_sim(crop, f"{subj_name} {relation} {obj_name}"), self.image_text_sim(crop, f"{obj_name} {relation} {subj_name}"), {"union_bbox": crop.getbbox()}

class EVAClipRelationScorer(EVAClipMixin, RelationScorerBackend):
    def _evaluate_relation(self, image, subj_bbox, obj_bbox, relation, subj_name, obj_name):
        crop = image.crop(tuple(union_bbox(subj_bbox, obj_bbox, self.config.stage3_margin_ratio, image.size)))
        return self.image_text_sim(crop, f"{subj_name} {relation} {obj_name}"), self.image_text_sim(crop, f"{obj_name} {relation} {subj_name}"), {"union_bbox": crop.getbbox()}

class ReITRRelationScorer(RelationScorerBackend):
    """RelTR/ReITR-style visual relationship scorer for stage 3."""

    CLASSES = [
        'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench',
        'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch',
        'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter',
        'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye',
        'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass',
        'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house',
        'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light',
        'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange',
        'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate',
        'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat',
        'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski',
        'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie',
        'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk',
        'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing',
        'wire', 'woman', 'zebra'
    ]
    REL_CLASSES = [
        '__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
        'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
        'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
        'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
        'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
        'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with'
    ]

    @classmethod
    def load_shared_runtime(cls, spec: BackendSpec, config: ExperimentConfig) -> Tuple[Dict[str, Any], float]:
        import sys
        import torch
        import torchvision.transforms as T
        from transformers import AutoModel, AutoTokenizer
        from types import SimpleNamespace

        start = time.perf_counter()
        code_dir = Path(spec.model_path or _default_reitr_model_path()).expanduser()
        checkpoint_path = spec.checkpoint_path or _default_reitr_checkpoint_path()
        if not checkpoint_path:
            raise ValueError("ReITR/RelTR backend requires --reitr-checkpoint-path or RELTR_CHECKPOINT_PATH.")
        if not code_dir.exists():
            raise FileNotFoundError(f"ReITR/RelTR code directory not found: {code_dir}")

        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        from models import build_model

        device = "cuda" if torch.cuda.is_available() and not config.use_cpu else "cpu"
        args = SimpleNamespace(
            lr_backbone=1e-5, dataset="vg", backbone="resnet50", dilation=False,
            position_embedding="sine", enc_layers=6, dec_layers=6, dim_feedforward=2048,
            hidden_dim=256, dropout=0.1, nheads=8, num_entities=100, num_triplets=200,
            pre_norm=False, aux_loss=True, device=device, resume=checkpoint_path,
            set_cost_class=1, set_cost_bbox=5, set_cost_giou=2, set_iou_threshold=0.7,
            bbox_loss_coef=5, giou_loss_coef=2, rel_loss_coef=1, eos_coef=0.1,
            return_interm_layers=False,
        )
        model, _, _ = build_model(args)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        model.to(device).eval()
        transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        text_model_path = _default_relation_text_embedding_model_path()
        text_tokenizer = AutoTokenizer.from_pretrained(text_model_path, padding_side="left", trust_remote_code=True)
        text_model = AutoModel.from_pretrained(
            text_model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        text_model.to(device).eval()
        load_time_ms = (time.perf_counter() - start) * 1000.0
        return ({
            "model": model,
            "transform": transform,
            "device": device,
            "text_tokenizer": text_tokenizer,
            "text_model": text_model,
        }, load_time_ms)

    def __init__(self, *args, shared_runtime: Optional[Dict[str, Any]] = None):
        super().__init__(*args)
        runtime = shared_runtime
        if runtime is None:
            runtime, self.model_load_time_ms = self.load_shared_runtime(self.spec, self.config)
        self.model = runtime["model"]
        self.transform = runtime["transform"]
        self.device = runtime["device"]
        self.text_tokenizer = runtime["text_tokenizer"]
        self.text_model = runtime["text_model"]
        self._relation_embedding_cache: Dict[str, Any] = {}

    def score_relations(self, image: Image.Image, item: ExperimentItem, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        node_map = {n["id"]: n for n in stage1_result.get("nodes", [])}
        entity_map = {e["id"]: e for e in item.scene_graph.get("objects", [])}
        triplets = self._predict_triplets(image)
        results = []
        for rel in item.scene_graph.get("relations", []):
            subj, obj = node_map.get(rel.get("subject")), node_map.get(rel.get("object"))
            if not subj or not obj or not subj.get("bbox") or not obj.get("bbox"):
                results.append({"subject": rel.get("subject"), "relation": rel.get("relation"), "object": rel.get("object"), "skipped": True, "skip_reason": "missing_localization"})
                continue
            subj_name = entity_map.get(rel.get("subject"), {}).get("name", rel.get("subject"))
            obj_name = entity_map.get(rel.get("object"), {}).get("name", rel.get("object"))
            orig_score, orig_match = self._score_triplets(triplets, subj_name, subj["bbox"], obj_name, obj["bbox"], rel["relation"])
            swap_score, swap_match = self._score_triplets(triplets, obj_name, obj["bbox"], subj_name, subj["bbox"], rel["relation"])
            results.append({
                "subject": rel["subject"],
                "relation": rel["relation"],
                "object": rel["object"],
                "original_score": orig_score,
                "swapped_score": swap_score,
                "delta": orig_score - swap_score,
                "swap_correct": orig_score > swap_score,
                "skipped": False,
                "matches": {"original": orig_match, "swapped": swap_match},
            })
        return {"backend": self.backend_id, "relations": results, "relation_score": safe_mean(e.get("original_score") for e in results if not e.get("skipped")), "swap_accuracy": safe_mean(1.0 if e.get("swap_correct") else 0.0 for e in results if e.get("swap_correct") is not None)}

    def _predict_triplets(self, image: Image.Image) -> List[Dict[str, Any]]:
        import torch

        def box_cxcywh_to_xyxy(x):
            x_c, y_c, w, h = x.unbind(1)
            return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=1)

        def rescale_bboxes(out_bbox, size):
            img_w, img_h = size
            return box_cxcywh_to_xyxy(out_bbox) * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=out_bbox.device)

        rgb = image.convert("RGB")
        inputs = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)

        rel_probs = outputs["rel_logits"].softmax(-1)[0, :, :-1]
        sub_probs = outputs["sub_logits"].softmax(-1)[0, :, :-1]
        obj_probs = outputs["obj_logits"].softmax(-1)[0, :, :-1]
        keep = torch.logical_and(
            rel_probs.max(-1).values > 0.3,
            torch.logical_and(sub_probs.max(-1).values > 0.3, obj_probs.max(-1).values > 0.3),
        )
        keep_queries = torch.nonzero(keep, as_tuple=True)[0]
        if keep_queries.numel() == 0:
            return []

        combined = rel_probs[keep_queries].max(-1).values * sub_probs[keep_queries].max(-1).values * obj_probs[keep_queries].max(-1).values
        order = torch.argsort(-combined)[:50]
        keep_queries = keep_queries[order]
        sub_boxes = rescale_bboxes(outputs["sub_boxes"][0, keep_queries], rgb.size).detach().cpu().tolist()
        obj_boxes = rescale_bboxes(outputs["obj_boxes"][0, keep_queries], rgb.size).detach().cpu().tolist()

        triplets = []
        for rank, idx in enumerate(keep_queries):
            rel_id = int(rel_probs[idx].argmax().item())
            sub_id = int(sub_probs[idx].argmax().item())
            obj_id = int(obj_probs[idx].argmax().item())
            score = float((rel_probs[idx, rel_id] * sub_probs[idx, sub_id] * obj_probs[idx, obj_id]).item())
            triplets.append({
                "subject": self.CLASSES[sub_id],
                "relation": self.REL_CLASSES[rel_id],
                "object": self.CLASSES[obj_id],
                "subject_bbox": sub_boxes[rank],
                "object_bbox": obj_boxes[rank],
                "score": score,
            })
        return triplets

    def _score_triplets(self, triplets: List[Dict[str, Any]], subj: str, subj_bbox: Sequence[int], obj: str, obj_bbox: Sequence[int], relation: str) -> Tuple[float, Optional[Dict[str, Any]]]:
        target_relation = _normalize_relation_label(relation)
        best_score = 0.0
        best_match = None
        for triplet in triplets:
            relation_similarity = self._relation_text_similarity(f"{triplet['subject']} {triplet['relation']} {triplet['object']}", f"{subj} {relation} {obj}")
            sub_iou = bbox_iou(subj_bbox, triplet["subject_bbox"])
            obj_iou = bbox_iou(obj_bbox, triplet["object_bbox"])
            loc_score = (sub_iou + obj_iou) / 2.0
            score = float(triplet["score"] * loc_score * relation_similarity)
            if score > best_score:
                best_score = score
                best_match = {
                    **triplet,
                    "subject_iou": sub_iou,
                    "object_iou": obj_iou,
                    "relation_similarity": relation_similarity,
                }
        return best_score, best_match

    def _relation_text_similarity(self, predicted: Any, target: Any) -> float:
        predicted_norm = _normalize_relation_label(predicted)
        target_norm = _normalize_relation_label(target)
        if not predicted_norm or not target_norm:
            return 0.0
        if predicted_norm == target_norm:
            return 1.0
        predicted_embedding = self._embed_relation_text(predicted_norm)
        target_embedding = self._embed_relation_text(target_norm)
        return max(0.0, float((predicted_embedding * target_embedding).sum().item()))

    def _embed_relation_text(self, text: str):
        import torch

        cached = self._relation_embedding_cache.get(text)
        if cached is not None:
            return cached

        inputs = self.text_tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            embeddings = _last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        embedding = embeddings[0].detach()
        self._relation_embedding_cache[text] = embedding
        return embedding

    def _evaluate_relation(self, image, subj_bbox, obj_bbox, relation, subj_name, obj_name):
        raise NotImplementedError("ReITRRelationScorer uses score_relations to cache per-image triplets.")

class VLMRelationScorer( _VisionLanguageMixin, RelationScorerBackend):
    def score_relations(self, image: Image.Image, item: ExperimentItem, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        entity_map = {e["id"]: e for e in item.scene_graph.get("objects", [])}
        results = []
        width, height = image.size
        for rel in item.scene_graph.get("relations", []):
            swapped_rel = {"subject": rel.get("object"), "relation": rel.get("relation"), "object": rel.get("subject")}
            orig_raw, orig_answer = _score_eval_yes_no_json(self.generate_text(image, build_relation_prompt(width, height, rel, entity_map)))
            swap_raw, swap_answer = _score_eval_yes_no_json(self.generate_text(image, build_relation_prompt(width, height, swapped_rel, entity_map)))
            cal = lambda s: calibrate_score(s, self.config.stage2_calibration, self.config.stage2_calibration_scale, self.config.stage2_calibration_bias)
            orig_score, swap_score = cal(orig_raw), cal(swap_raw)
            results.append({
                "subject": rel["subject"],
                "relation": rel["relation"],
                "object": rel["object"],
                "original_score": orig_score,
                "swapped_score": swap_score,
                "delta": orig_score - swap_score,
                "swap_correct": orig_score > swap_score,
                "answers": {"original": orig_answer, "swapped": swap_answer},
                "skipped": False,
            })
        return {"backend": self.backend_id, "relations": results, "relation_score": safe_mean(e.get("original_score") for e in results if not e.get("skipped")), "swap_accuracy": safe_mean(1.0 if e.get("swap_correct") else 0.0 for e in results if e.get("swap_correct") is not None)}

    def _evaluate_relation(self, image, subj_bbox, obj_bbox, relation, subj_name, obj_name):
        raise NotImplementedError("VLMRelationScorer uses score_relations with eval-pipeline prompts.")


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

class SkippedAttributeScorer(AttributeScorerBackend):
    def score_attributes(self, image: Image.Image, item: ExperimentItem, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        results = []
        for entity in item.scene_graph.get("objects", []):
            for attr in entity.get("attributes", []):
                results.append({
                    "id": entity.get("id"),
                    "name": entity.get("name"),
                    "attribute": attr,
                    "skipped": True,
                    "skip_reason": "stage2_skipped",
                })
        return {
            "backend": self.backend_id,
            "crop_size": None,
            "attributes": results,
            "binding_score": None,
            "skipped": True,
            "skip_reason": "stage2_skipped",
        }

    def _evaluate_attribute(self, crop, entity_name, attribute):
        raise NotImplementedError("SkippedAttributeScorer bypasses stage 2 attribute scoring.")

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
    if runtime_key[0] == "eva-clip":
        return EVAClipMixin.load_shared_runtime(spec, config)
    if runtime_key[0] == "blip2":
        return BLIP2Mixin.load_shared_runtime(spec, config)
    if runtime_key[0] == "reitr":
        return ReITRRelationScorer.load_shared_runtime(spec, config)
    if runtime_key[0] == "qwen-hf":
        return _VisionLanguageMixin.load_shared_runtime(spec, config)
    if runtime_key[0] == "qwen-vllm":
        return _QwenVLLMMixin.load_shared_runtime(spec, config)
    return None, 0.0

def build_backend(backend_id: str, spec: BackendSpec, config: ExperimentConfig, shared_runtimes: Optional[Dict[Tuple[Any, ...], Dict[str, Any]]] = None) -> Any:
    k = (spec.kind or "").lower()
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
        case "E1":
            if k in {"grounding", "grounding-dino", "hf-grounding"}:
                return DinoClipNodeDetector(backend_id, spec, config)
            return UnavailableBackend(backend_id, spec, config)
        case "V1": return QwenNodeDetectorVLLM(backend_id, spec, config, shared_runtime=shared_runtime) if config.use_vllm else QwenNodeDetector(backend_id, spec, config, shared_runtime=shared_runtime) if "qwen" in k else UnavailableBackend(backend_id, spec, config)
        case "E2":
            if k in {"eva", "eva-clip", "evaclip"}:
                return EVAClipAttributeScorer(backend_id, spec, config, shared_runtime=shared_runtime)
            if k in {"blip2", "blip-2"}:
                return BLIP2AttributeScorer(backend_id, spec, config, shared_runtime=shared_runtime)
            return SigLIPAttributeScorer(backend_id, spec, config, shared_runtime=shared_runtime) if "siglip" in k else UnavailableBackend(backend_id, spec, config)
        case "V2":
            if k in {"qwen-molmopoint", "qwen-molmo-point", "molmopoint-qwen", "molmo-point-qwen"}:
                return QwenMolmoPointAttributeClassifierVLLM(backend_id, spec, config, shared_runtime=shared_runtime) if config.use_vllm else VLMMolmoPointAttributeScorer(backend_id, spec, config, shared_runtime=shared_runtime)
            return QwenAttributeClassifierVLLM(backend_id, spec, config, shared_runtime=shared_runtime) if config.use_vllm else VLMAttributeScorer(backend_id, spec, config, shared_runtime=shared_runtime) if k in {"llava", "llava-next", "qwen", "qwen-vl"} else UnavailableBackend(backend_id, spec, config)
        case "S2": return SkippedAttributeScorer(backend_id, spec, config) if k in {"skip", "skipped", "none"} else UnavailableBackend(backend_id, spec, config)
        case "E3":
            if k in {"eva", "eva-clip", "evaclip"}:
                return EVAClipRelationScorer(backend_id, spec, config, shared_runtime=shared_runtime)
            if k in {"reitr", "reltr"}:
                return ReITRRelationScorer(backend_id, spec, config, shared_runtime=shared_runtime)
            return SigLIPRelationScorer(backend_id, spec, config, shared_runtime=shared_runtime) if "siglip" in k else UnavailableBackend(backend_id, spec, config)
        case "V3": return QwenRelationScorerVLLM(backend_id, spec, config, shared_runtime=shared_runtime) if config.use_vllm else VLMRelationScorer(backend_id, spec, config, shared_runtime=shared_runtime) if "qwen" in k else UnavailableBackend(backend_id, spec, config)
    return UnavailableBackend(backend_id, spec, config)

# --- Geometry & Metric Utils ---
CONTRASTIVE_ATTRIBUTE_TEMPERATURE = 25.0
COUNT_WORDS = {
    "zero": 0,
    "no": 0,
    "single": 1,
    "one": 1,
    "individual": 1,
    "sole": 1,
    "two": 2,
    "pair": 2,
    "couple": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
COUNT_ATTRIBUTE_EXCLUSIONS = {
    "colored",
    "colorful",
    "multicolored",
    "unicolored",
    "single-colored",
    "two-colored",
    "three-colored",
}

def _score_eval_yes_no_json(raw_text: str) -> Tuple[float, str]:
    evaluation = extract_json(raw_text)
    answer = normalize_answer(evaluation.get("satisfies") if evaluation else raw_text)
    return (1.0 if answer == "yes" else 0.0), answer

def parse_count_attribute(attribute: Any) -> Optional[int]:
    text = str(attribute).strip().lower()
    if not text:
        return None
    if text == "single/one/individual/sole":
        return 1
    if any(term in text for term in COUNT_ATTRIBUTE_EXCLUSIONS):
        return None

    numeric_match = re.search(r"\b(\d{1,2})\b", text)
    if numeric_match:
        return int(numeric_match.group(1))

    tokens = [token for token in re.split(r"[^a-z]+", text) if token]
    for token in tokens:
        if token in COUNT_WORDS:
            return COUNT_WORDS[token]
    return None

def build_contrastive_attribute_prompts(entity_name: str, attribute: str) -> Tuple[str, str]:
    entity_name = str(entity_name).strip()
    attribute = str(attribute).strip()
    return (
        f"{attribute} {entity_name}",
        f"not {attribute} {entity_name}",
    )

def contrastive_binary_score(positive_score: float, negative_score: float, temperature: float = CONTRASTIVE_ATTRIBUTE_TEMPERATURE) -> float:
    logit = max(-60.0, min(60.0, temperature * (positive_score - negative_score)))
    return 1.0 / (1.0 + math.exp(-logit))

def score_contrastive_attribute(sim_fn, crop: Image.Image, entity_name: str, attribute: str) -> Tuple[float, Dict[str, Any]]:
    positive_prompt, negative_prompt = build_contrastive_attribute_prompts(entity_name, attribute)
    positive_score = float(sim_fn(crop, positive_prompt))
    negative_score = float(sim_fn(crop, negative_prompt))
    score = contrastive_binary_score(positive_score, negative_score)
    return score, {
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "contrastive_temperature": CONTRASTIVE_ATTRIBUTE_TEMPERATURE,
    }

def _node_from_eval_object_response(
    raw_text: str,
    entity: Dict[str, Any],
    image_size: Tuple[int, int],
    confidence_threshold: float,
) -> Dict[str, Any]:
    evaluation = extract_json(raw_text)
    visible = normalize_visible(evaluation.get("visible") if evaluation else None)
    bbox = normalize_bbox(evaluation.get("bbox") if evaluation else None)
    clamped_bbox = clamp_bbox(bbox, *image_size) if visible and bbox else None
    confidence = 1.0 if visible else 0.0
    passed = bool(clamped_bbox) and confidence >= confidence_threshold
    return {
        "id": entity.get("id"),
        "name": entity.get("name"),
        "bbox": clamped_bbox if passed else None,
        "confidence": confidence,
        "passed": passed,
        "score": 1.0 if passed else 0.0,
    }

def _normalize_relation_label(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).replace("_", " ").replace("-", " ").strip().lower())

def _last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[range(batch_size), sequence_lengths]

def _resolve_checkpoint_path(path_or_url: Optional[str]) -> Optional[str]:
    if not path_or_url:
        return None
    if path_or_url.startswith("https://huggingface.co/"):
        from huggingface_hub import hf_hub_download

        match = re.match(r"https://huggingface\.co/([^/]+/[^/]+)/blob/([^/]+)/(.+)", path_or_url)
        if match:
            repo_id, revision, filename = match.groups()
            return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    return path_or_url

def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def safe_mean(v: Iterable) -> Optional[float]: 
    cleaned = [float(x) for x in v if x is not None]
    return sum(cleaned) / len(cleaned) if cleaned else None

def build_pipeline_permutations(
    stage1_variants: Sequence[str] = STAGE1_VARIANTS,
    stage2_variants: Sequence[str] = DEFAULT_STAGE2_VARIANTS,
    stage3_variants: Sequence[str] = STAGE3_VARIANTS,
) -> List[Tuple[str, str, str]]:
    return [(s1, s2, s3) for s1 in stage1_variants for s2 in stage2_variants for s3 in stage3_variants]

def normalize_weights(weights: StageWeights) -> Dict[str, Dict[str, float]]:
    raw = {"node": weights.node, "attribute": weights.attribute, "relation": weights.relation}
    total = sum(raw.values())
    normalized = {key: (value / total if total else 0.0) for key, value in raw.items()}
    return {"raw": raw, "normalized": normalized}

def compose_score(scores: Mapping[str, Optional[float]], weights: Mapping[str, float]) -> Dict[str, Any]:
    active = {key: value for key, value in scores.items() if value is not None}
    active_weight_total = sum(weights.get(key, 0.0) for key in active)
    if not active or active_weight_total <= 0:
        return {"score": None, "active_weights": {}}
    active_weights = {key: weights.get(key, 0.0) / active_weight_total for key in active}
    return {
        "score": sum(active[key] * active_weights[key] for key in active),
        "active_weights": active_weights,
    }

def compute_correlation_report(
    rows_by_perm: Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]],
    labels: Optional[Sequence[Mapping[str, Any]]],
    label_config: LabelConfig,
) -> Optional[Dict[str, Any]]:
    if labels is None:
        if not label_config.path or not Path(label_config.path).exists():
            return None
        labels = load_json_or_jsonl(label_config.path)

    label_map = {
        str(row.get(label_config.key_field)): row.get(label_config.score_field)
        for row in labels
        if row.get(label_config.key_field) is not None and row.get(label_config.score_field) is not None
    }
    if not label_map:
        return None

    grouped = rows_by_perm if isinstance(rows_by_perm, Mapping) else {"all": rows_by_perm}
    report: Dict[str, Any] = {}
    for perm, rows in grouped.items():
        pred, gold = [], []
        for row in rows:
            key = str(row.get(label_config.result_key_field))
            if key in label_map and row.get("final_score") is not None:
                pred.append(float(row["final_score"]))
                gold.append(float(label_map[key]))
        if not pred:
            continue
        pearson = None
        if len(pred) >= 2:
            pred_arr = np.asarray(pred, dtype=float)
            gold_arr = np.asarray(gold, dtype=float)
            if pred_arr.std() > 0 and gold_arr.std() > 0:
                pearson = float(np.corrcoef(pred_arr, gold_arr)[0, 1])
        report[perm] = {"n": len(pred), "pearson": pearson}
    return report or None

def invert_relation(relation: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "subject": relation.get("object"),
        "relation": relation.get("relation"),
        "object": relation.get("subject"),
    }

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
    if config.skip_indices:
        skip_indices = set(config.skip_indices)
        items = [item for item in items if item.prompt_index not in skip_indices]

    return items

def run_ablation_experiment(config: ExperimentConfig, items=None, backends=None) -> Dict[str, Any]:
    import torch
    norm_wts = normalize_weights(config.weights)["normalized"]
    items = items or load_experiment_items(config)

    # Determine which backends to use based on config or defaults to all
    if hasattr(config, 'selected_backends') and config.selected_backends:
        selected_backends = config.selected_backends
    else:
        selected_backends = set(STAGE1_VARIANTS + DEFAULT_STAGE2_VARIANTS + STAGE3_VARIANTS)

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
            composed = compose_score(sc, norm_wts)

            rows_by_perm[perm].append({
                "image_id": item.image_id, "prompt": item.prompt, "permutation": perm, "final_score": composed["score"],
                "st1_res": st1_cache[b1]["res"], 
                "st2_res": st2_cache[(b1, b2)]["res"], 
                "st3_res": st3_cache[(b1, b3)]["res"], 
                "latency_ms": {"total": st1_cache[b1]["lat"] + st2_cache[(b1, b2)]["lat"] + st3_cache[(b1, b3)]["lat"]}
            }) # Condensed dict for brevity

    aggregate_matrix = [
        {
            "permutation": perm,
            "n": len(rows),
            "average_final_score": safe_mean(row.get("final_score") for row in rows),
        }
        for perm, rows in rows_by_perm.items()
    ]
    latency_report = {
        perm: {"average_total_ms": safe_mean(row.get("latency_ms", {}).get("total") for row in rows)}
        for perm, rows in rows_by_perm.items()
    }
    relation_swap_report = {
        perm: {
            "average_swap_accuracy": safe_mean(
                row.get("st3_res", {}).get("swap_accuracy") for row in rows
            )
        }
        for perm, rows in rows_by_perm.items()
    }
    correlation_report = compute_correlation_report(rows_by_perm, None, config.label_config)

    return {
        "config": serialize_config(config),
        "items_total": len(items),
        "permutations": rows_by_perm,
        "aggregate_matrix": aggregate_matrix,
        "latency_report": latency_report,
        "relation_swap_report": relation_swap_report,
        "correlation_report": correlation_report,
    }

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
    p.add_argument("--skip-indices", default="", help="Comma-separated prompt/image indices to exclude.")
    p.add_argument("--human-score-file", default=None)
    p.add_argument("--label-key-field", default="image_id")
    p.add_argument("--label-score-field", default="score")
    p.add_argument("--result-key-field", default="image_id")
    p.add_argument("--weight-node", type=float, default=0.3)
    p.add_argument("--weight-attribute", type=float, default=0.3)
    p.add_argument("--weight-relation", type=float, default=0.3)
    p.add_argument("--node-confidence-threshold", type=float, default=0.2)
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
    p.add_argument("--max-text-length", type=int, default=64, help="Max text length for text/image embedding models (default: 64)")
    p.add_argument("--torch-cuda-mem-frac", type=float, default=0.8, help="Fraction of GPU memory to use (for device_map='auto')")

    # Condensed repetitive backend argument parsing
    for b in ("e1", "v1", "e2", "v2", "e3", "v3"):
        p.add_argument(f"--{b}-backend-kind", default=None) # was "mock")

    for m in ("eupe", "qwen", "siglip", "eva-clip", "blip2", "molmopoint", "reitr", "llava"):
        p.add_argument(f"--{m}-model-path", default=None)
        p.add_argument(f"--{m}-checkpoint-path", default=None)

    return p
def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    none_vals = {None, "None"}
    get = lambda name, default=None: getattr(args, name, default)
    backend_kind = lambda name: get(f"{name}_backend_kind", "mock")
    model_path = lambda name: get(f"{name}_model_path", None)
    checkpoint_path = lambda name: get(f"{name}_checkpoint_path", None)
    e2_kind = backend_kind("e2")
    e3_kind = backend_kind("e3")
    return ExperimentConfig(
        output_dir=args.output_dir,
        prompts_file=None if get("prompts_file") in none_vals else get("prompts_file"),
        sg_file=None if get("sg_file") in none_vals else get("sg_file"),
        images_dir=args.images_dir,
        image_pattern=get("image_pattern", "{index:04d}-{generation}.png"),
        generation=get("generation", 1),
        start_idx=get("start_idx", 0),
        end_idx=get("end_idx", None),
        limit=get("limit", None),
        skip_indices=tuple(int(x.strip()) for x in get("skip_indices", "").split(",") if x.strip()),
        weights=StageWeights(get("weight_node", 0.3), get("weight_attribute", 0.3), get("weight_relation", 0.3)),
        node_confidence_threshold=get("node_confidence_threshold", 0.2),
        node_nms_threshold=get("node_nms_threshold", 0.3),
        stage2_crop_size=get("stage2_crop_size", 384),
        stage2_calibration=get("stage2_calibration", "clip"),
        stage2_calibration_scale=get("stage2_calibration_scale", 1.0),
        stage2_calibration_bias=get("stage2_calibration_bias", 0.0),
        stage3_margin_ratio=get("stage3_margin_ratio", 0.1),
        include_model_load_time=get("include_model_load_time", False),
        label_config=LabelConfig(
            path=None if get("human_score_file") in none_vals else get("human_score_file"),
            key_field=get("label_key_field", "image_id"),
            score_field=get("label_score_field", "score"),
            result_key_field=get("result_key_field", "image_id"),
        ),
        backend_specs={
            "E1": BackendSpec(backend_kind("e1"), model_path("eupe"), checkpoint_path("eupe")),
            "V1": BackendSpec(backend_kind("v1"), model_path("qwen"), checkpoint_path("qwen")),
            "E2": BackendSpec(
                e2_kind,
                model_path("eva_clip") if e2_kind in {"eva", "eva-clip", "evaclip"} else model_path("blip2") if e2_kind in {"blip2", "blip-2"} else model_path("siglip"),
                checkpoint_path("eva_clip") if e2_kind in {"eva", "eva-clip", "evaclip"} else checkpoint_path("blip2") if e2_kind in {"blip2", "blip-2"} else checkpoint_path("siglip"),
            ),
            "V2": BackendSpec(backend_kind("v2"), model_path("qwen"), checkpoint_path("qwen")),
            "S2": BackendSpec("skip"),
            "E3": BackendSpec(
                e3_kind,
                model_path("reitr") if e3_kind in {"reitr", "reltr"} else model_path("eva_clip") if e3_kind in {"eva", "eva-clip", "evaclip"} else model_path("siglip"),
                checkpoint_path("reitr") if e3_kind in {"reitr", "reltr"} else checkpoint_path("eva_clip") if e3_kind in {"eva", "eva-clip", "evaclip"} else checkpoint_path("siglip"),
            ),
            "V3": BackendSpec(backend_kind("v3"), model_path("qwen"), checkpoint_path("qwen")),
        },
        selected_backends=set(get("backends").split(",")) if get("backends") else None,
        use_cpu=get("cpu", False),
        low_vram=get("low_vram", False),
        use_vllm=get("use_vllm", False),
        max_text_length=get("max_text_length", 64),
        torch_cuda_mem_frac=get("torch_cuda_mem_frac", 0.8),
        vllm_api_base=get("vllm_api_base", "http://127.0.0.1:8000/v1"),
        vllm_api_key=get("vllm_api_key", None),
        vllm_temperature=get("vllm_temperature", None),
        vllm_max_tokens=get("vllm_max_tokens", None),
        vllm_yes_no_max_tokens=get("vllm_yes_no_max_tokens", None),
        molmopoint_model_path=model_path("molmopoint"),
        molmopoint_checkpoint_path=checkpoint_path("molmopoint"),
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
