import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT))

processing = types.ModuleType("modules.processing")
processing.apply_batch_results = lambda *args, **kwargs: None
processing.build_attribute_prompt = lambda *args, **kwargs: ""
processing.build_object_prompt = lambda *args, **kwargs: ""
processing.build_relation_prompt = lambda *args, **kwargs: ""
processing.image_path_from_pattern = lambda *args, **kwargs: ""
processing.load_json_or_jsonl = lambda *args, **kwargs: []
processing.summarize_results = lambda *args, **kwargs: {}
processing.extract_scene_graph = lambda *args, **kwargs: {}
sys.modules["modules"] = types.ModuleType("modules")
sys.modules["modules.processing"] = processing

from src.eval.ablation import (
    BackendSpec,
    ExperimentConfig,
    LabelConfig,
    SigLIPMixin,
    StageWeights,
    VLMAttributeScorer,
    VLMRelationScorer,
    QwenNodeDetector,
    QwenNodeDetectorVLLM,
    QwenAttributeClassifierVLLM,
    QwenRelationScorerVLLM,
    _QwenVLLMMixin,
    _VisionLanguageMixin,
    build_backend,
)


def make_config(use_vllm: bool = False) -> ExperimentConfig:
    return ExperimentConfig(
        output_dir="tmp",
        prompts_file=None,
        sg_file=None,
        images_dir="tmp",
        image_pattern="{index:04d}-{generation}.png",
        generation=1,
        start_idx=0,
        end_idx=None,
        limit=None,
        skip_indices=(),
        weights=StageWeights(1.0, 1.0, 1.0),
        node_confidence_threshold=0.5,
        node_nms_threshold=0.3,
        stage2_crop_size=384,
        stage2_calibration="clip",
        stage2_calibration_scale=1.0,
        stage2_calibration_bias=0.0,
        stage3_margin_ratio=0.1,
        include_model_load_time=False,
        label_config=LabelConfig(path=None),
        backend_specs={
            "V1": BackendSpec("qwen", "qwen-a", None),
            "V2": BackendSpec("qwen", "qwen-a", None),
            "V3": BackendSpec("qwen", "qwen-a", None),
            "E2": BackendSpec("siglip", "siglip-a", None),
            "E3": BackendSpec("siglip", "siglip-a", None),
        },
        selected_backends=None,
        use_cpu=True,
        low_vram=False,
        use_vllm=use_vllm,
        max_text_length=64,
        torch_cuda_mem_frac=0.8,
    )


def test_qwen_transformers_runtime_sharing() -> None:
    original = _VisionLanguageMixin.load_shared_runtime
    calls = []

    def fake_loader(cls, spec, config):
        calls.append((spec.model_path, spec.checkpoint_path))
        return ({
            "proc": object(),
            "model": object(),
            "_cuda_available": False,
            "_batch_size": 1,
            "_max_new_tokens": 256,
        }, 1.0)

    _VisionLanguageMixin.load_shared_runtime = classmethod(fake_loader)
    try:
        config = make_config(use_vllm=False)
        shared_runtimes = {}
        v1 = build_backend("V1", config.backend_specs["V1"], config, shared_runtimes=shared_runtimes)
        v2 = build_backend("V2", config.backend_specs["V2"], config, shared_runtimes=shared_runtimes)
        v3 = build_backend("V3", config.backend_specs["V3"], config, shared_runtimes=shared_runtimes)

        assert isinstance(v1, QwenNodeDetector)
        assert isinstance(v2, VLMAttributeScorer)
        assert isinstance(v3, VLMRelationScorer)
        assert v1.model is v2.model is v3.model
        assert v1.proc is v2.proc is v3.proc
        assert len(calls) == 1
    finally:
        _VisionLanguageMixin.load_shared_runtime = original


def test_siglip_runtime_sharing() -> None:
    original = SigLIPMixin.load_shared_runtime
    calls = []

    def fake_loader(cls, spec, config):
        calls.append((spec.model_path, spec.checkpoint_path))
        return ({
            "proc": object(),
            "model": object(),
            "use_siglip2": True,
            "_device_map_used": False,
        }, 1.0)

    SigLIPMixin.load_shared_runtime = classmethod(fake_loader)
    try:
        config = make_config(use_vllm=False)
        shared_runtimes = {}
        e2 = build_backend("E2", config.backend_specs["E2"], config, shared_runtimes=shared_runtimes)
        e3 = build_backend("E3", config.backend_specs["E3"], config, shared_runtimes=shared_runtimes)

        assert e2.model is e3.model
        assert e2.proc is e3.proc
        assert len(calls) == 1
    finally:
        SigLIPMixin.load_shared_runtime = original


def test_qwen_vllm_runtime_sharing() -> None:
    original = _QwenVLLMMixin.load_shared_runtime
    calls = []

    def fake_loader(cls, spec, config):
        calls.append((spec.model_path, spec.checkpoint_path))
        return ({
            "llm": object(),
            "sampling_params": object(),
        }, 1.0)

    _QwenVLLMMixin.load_shared_runtime = classmethod(fake_loader)
    try:
        config = make_config(use_vllm=True)
        shared_runtimes = {}
        v1 = build_backend("V1", config.backend_specs["V1"], config, shared_runtimes=shared_runtimes)
        v2 = build_backend("V2", config.backend_specs["V2"], config, shared_runtimes=shared_runtimes)
        v3 = build_backend("V3", config.backend_specs["V3"], config, shared_runtimes=shared_runtimes)

        assert isinstance(v1, QwenNodeDetectorVLLM)
        assert isinstance(v2, QwenAttributeClassifierVLLM)
        assert isinstance(v3, QwenRelationScorerVLLM)
        assert v1.llm is v2.llm is v3.llm
        assert v1.sampling_params is v2.sampling_params is v3.sampling_params
        assert len(calls) == 1
    finally:
        _QwenVLLMMixin.load_shared_runtime = original


def test_different_model_specs_do_not_share() -> None:
    original = _VisionLanguageMixin.load_shared_runtime
    calls = []

    def fake_loader(cls, spec, config):
        calls.append((spec.model_path, spec.checkpoint_path))
        return ({
            "proc": object(),
            "model": object(),
            "_cuda_available": False,
            "_batch_size": 1,
            "_max_new_tokens": 256,
        }, 1.0)

    _VisionLanguageMixin.load_shared_runtime = classmethod(fake_loader)
    try:
        config = make_config(use_vllm=False)
        shared_runtimes = {}
        v1 = build_backend("V1", BackendSpec("qwen", "qwen-a", None), config, shared_runtimes=shared_runtimes)
        v2 = build_backend("V2", BackendSpec("qwen", "qwen-b", None), config, shared_runtimes=shared_runtimes)

        assert v1.model is not v2.model
        assert len(calls) == 2
    finally:
        _VisionLanguageMixin.load_shared_runtime = original


if __name__ == "__main__":
    test_qwen_transformers_runtime_sharing()
    test_siglip_runtime_sharing()
    test_qwen_vllm_runtime_sharing()
    test_different_model_specs_do_not_share()
    print("Shared model runtime tests passed.")
