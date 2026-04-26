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

class StageBackend(ABC):
    def __init__(self, backend_id: str, spec: BackendSpec, config: ExperimentConfig):
        self.backend_id, self.spec, self.config, self.model_load_time_ms = backend_id, spec, config, 0.0

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
        from transformers import AutoModel, AutoModelForImageTextToText, AutoModelForZeroShotObjectDetection, AutoProcessor
        return AutoProcessor, AutoModel, AutoModelForImageTextToText, AutoModelForZeroShotObjectDetection

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
        
        # Load DINO for box proposals (falls back to generic DINO if no model_path specified)
        dino_path = self.spec.model_path or "IDEA-Research/grounding-dino-base"
        self.dino_proc = AutoProcessor.from_pretrained(dino_path)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_path)

        # Load CLIP for text/image matching
        clip_path = "openai/clip-vit-base-patch32"
        self.clip_proc = CLIPProcessor.from_pretrained(clip_path)
        self.clip_model = CLIPModel.from_pretrained(clip_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dino_model.to(self.device)
        self.clip_model.to(self.device)
        self.dino_model.eval()
        self.clip_model.eval()
        
        self.model_load_time_ms = (time.perf_counter() - start) * 1000.0

    def detect_nodes(self, image: Image.Image, item: ExperimentItem) -> Dict[str, Any]:
        import torch
        # breakpoint()
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
    def __init__(self, *args):
        super().__init__(*args)
        self.proc, self.model = self._load_model(self._load_components()[2])

    def generate_text(self, image: Image.Image, prompt: str) -> str:
        import torch
        batch = self._to_device(self.proc.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"), self.model)
        with torch.inference_mode(): gen = self.model.generate(**batch, max_new_tokens=128, do_sample=False)
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

    def detect_nodes(self, image: Image.Image, item: ExperimentItem) -> Dict[str, Any]:
        nodes = []
        for entity in item.scene_graph.get("objects", []):
            prompt = f"Return JSON only.\nLocate '{entity.get('name')}' in the image using normalized coordinates [0,1000]. If absent return {{\"boxes\": [], \"confidence\": 0.0}}."
            raw = self.generate_text(image, prompt)
            try: parsed = json.loads(raw[raw.find("{"):raw.rfind("}")+1]) if "{" in raw else {}
            except json.JSONDecodeError: parsed = {}
            boxes = parse_stage1_localization(json.dumps(parsed.get("boxes", [])), image.size)
            conf = float(parsed.get("confidence", 1.0 if boxes else 0.0))
            passed = bool(boxes) and conf >= self.config.node_confidence_threshold
            nodes.append({"id": entity.get("id"), "name": entity.get("name"), "bbox": boxes[0] if passed else None, "confidence": conf, "passed": passed, "score": 1.0 if passed else 0.0})
        return {"backend": self.backend_id, "nodes": nodes, "fidelity_score": safe_mean(n["score"] for n in nodes)}

class SigLIPMixin(_TransformersBackendMixin):
    def __init__(self, *args):
        super().__init__(*args)
        if self.spec.model_path: self.spec = BackendSpec(self.spec.kind, resolve_siglip_model_path(self.spec.model_path), self.spec.checkpoint_path)
        self.proc, self.model = self._load_model(self._load_components()[1])

    def image_text_sim(self, image: Image.Image, text: str) -> float:
        import torch
        batch = self._to_device(self.proc(images=image, text=[text], return_tensors="pt", padding=True), self.model)
        with torch.inference_mode():
            if hasattr(self.model, "get_image_features"):
                img_f, txt_f = self.model.get_image_features(pixel_values=batch["pixel_values"]), self.model.get_text_features(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
            else:
                out = self.model(**batch); img_f, txt_f = out.image_embeds, out.text_embeds
        return float((torch.nn.functional.normalize(img_f, dim=-1) @ torch.nn.functional.normalize(txt_f, dim=-1).transpose(0, 1))[0, 0].item())

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

def build_backend(backend_id: str, spec: BackendSpec, config: ExperimentConfig) -> Any:
    k = spec.kind.lower()
    if k == "mock": return MockPipeline(backend_id, spec, config)
    match backend_id:
        case "E1": return DinoClipNodeDetector(backend_id, spec, config) if k in {"grounding", "grounding-dino", "hf-grounding"} else UnavailableBackend(backend_id, spec, config)
        case "V1": return QwenNodeDetector(backend_id, spec, config) if "qwen" in k else UnavailableBackend(backend_id, spec, config)
        case "E2": return SigLIPAttributeScorer(backend_id, spec, config) if "siglip" in k else UnavailableBackend(backend_id, spec, config)
        case "V2": return VLMAttributeScorer(backend_id, spec, config) if k in {"llava", "llava-next", "qwen", "qwen-vl"} else UnavailableBackend(backend_id, spec, config)
        case "E3": return SigLIPRelationScorer(backend_id, spec, config) if "siglip" in k else UnavailableBackend(backend_id, spec, config)
        case "V3": return VLMRelationScorer(backend_id, spec, config) if "qwen" in k else UnavailableBackend(backend_id, spec, config)
    return UnavailableBackend(backend_id, spec, config)

# --- Geometry & Metric Utils ---
def safe_mean(v: Iterable) -> Optional[float]: 
    cleaned = [float(x) for x in v if x is not None]
    return sum(cleaned) / len(cleaned) if cleaned else None

def clamp_bbox(b: Sequence[float], w: int, h: int) -> List[int]: return [max(0, min(int(round(b[0])), w-1)), max(0, min(int(round(b[1])), h-1)), max(1, min(int(round(b[2])), w)), max(1, min(int(round(b[3])), h))]
def normalized_bbox_to_pixel(b: Sequence[float], w: int, h: int) -> List[int]: return clamp_bbox([w*b[0]/1000.0, h*b[1]/1000.0, w*b[2]/1000.0, h*b[3]/1000.0], w, h)
def parse_stage1_localization(raw: str, size: Tuple[int, int]) -> List[List[int]]: return [normalized_bbox_to_pixel(b["bbox"] if isinstance(b, dict) else b, *size) for b in (json.loads(raw).get("boxes", []) if "{" in raw else []) if isinstance(b, (list, dict))]
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
    """Load experiment items from prompts file and image directory."""
    items = []

    if config.prompts_file:
        try:
            prompts_data = load_json_or_jsonl(args.prompts_file)
        except Exception as e:
            raise Exception("Couldn't load prompts file.") 

        for idx in tqdm(range(len(prompts_data))):
            entry = prompts_data[idx]
            scene_graph = extract_scene_graph(entry["meta_prompt"]["prompt"]) if "meta_prompt" in entry else None
            prompt = entry["prompt"]

            for i in range(args.generation):
                if args.prompts_file:
                    image_path = image_path_from_pattern(args.image_pattern, config.images_dir, idx, i + 1)
                else: 
                    image_path = os.path.join(images_dir, scene_graph["filename"])
                if not os.path.exists(image_path):
                    print("Warning: didn't find image at path ", image_path)
                    continue

                image_id = os.path.basename(image_path).split(".png")[0]

                items.append(ExperimentItem(
                            prompt_index=idx,
                            image_id=str(image_id),
                            prompt=str(prompt),
                            image_path=str(image_path),
                            scene_graph=scene_graph
                        ))

    # Apply filtering
    if config.start_idx > 0:
        items = items[config.start_idx:]
    if config.end_idx is not None:
        items = items[:config.end_idx]
    if config.limit is not None:
        items = items[:config.limit]

    return items

def run_ablation_experiment(config: ExperimentConfig, items=None, backends=None) -> Dict[str, Any]:
    wt = {"node": config.weights.node, "attribute": config.weights.attribute, "relation": config.weights.relation}
    norm_wts = {k: v / sum(wt.values()) for k, v in wt.items()}
    items = items or load_experiment_items(config)
    bm = backends or {b: build_backend(b, config.backend_specs[b], config) for b in STAGE1_VARIANTS + STAGE2_VARIANTS + STAGE3_VARIANTS}

    rows_by_perm = {f"{s1}-{s2}-{s3}": [] for s1 in STAGE1_VARIANTS for s2 in STAGE2_VARIANTS for s3 in STAGE3_VARIANTS}
    time_call = lambda f, *args: (lambda st=time.perf_counter(), res=f(*args): (res, (time.perf_counter() - st) * 1000.0))()

    for item in items:
        with Image.open(item.image_path) as img_h: img = img_h.convert("RGB")
        st1_cache = {b: dict(zip(["res", "lat"], time_call(bm[b].detect_nodes, img, item))) for b in STAGE1_VARIANTS}
        st2_cache = {(b1, b2): dict(zip(["res", "lat"], time_call(bm[b2].score_attributes, img, item, st1_cache[b1]["res"]))) for b1 in STAGE1_VARIANTS for b2 in STAGE2_VARIANTS}
        st3_cache = {(b1, b3): dict(zip(["res", "lat"], time_call(bm[b3].score_relations, img, item, st1_cache[b1]["res"]))) for b1 in STAGE1_VARIANTS for b3 in STAGE3_VARIANTS}

        for perm in rows_by_perm.keys():
            b1, b2, b3 = perm.split("-")
            sc = {"node": st1_cache[b1]["res"].get("fidelity_score"), "attribute": st2_cache[(b1, b2)]["res"].get("binding_score"), "relation": st3_cache[(b1, b3)]["res"].get("relation_score")}
            act_wts = {k: norm_wts[k] / sum(norm_wts[x] for x in sc if sc[x] is not None) for k in sc if sc[k] is not None} if any(v is not None for v in sc.values()) else {}
            
            rows_by_perm[perm].append({
                "image_id": item.image_id, "prompt": item.prompt, "permutation": perm, "final_score": sum(sc[k] * act_wts[k] for k in act_wts),
                "latency_ms": {"total": st1_cache[b1]["lat"] + st2_cache[(b1, b2)]["lat"] + st3_cache[(b1, b3)]["lat"]}
            }) # Condensed dict for brevity
            
    return {"config": asdict(config), "items_total": len(items), "permutations": rows_by_perm}

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
            "V2": BackendSpec(args.v2_backend_kind, args.llava_model_path, args.llava_checkpoint_path),
            "E3": BackendSpec(args.e3_backend_kind, args.siglip_model_path, args.siglip_checkpoint_path),
            "V3": BackendSpec(args.v3_backend_kind, args.qwen_model_path, args.qwen_checkpoint_path),
        },
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
    report = run_ablation_experiment(config)
    write_experiment_outputs(report, config.output_dir)
