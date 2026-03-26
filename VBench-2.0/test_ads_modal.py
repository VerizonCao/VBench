"""
Ads Video Evaluation on Modal — Concurrent Execution.

Runs THREE tiers:
  1. VBench 2.0 GPU dimensions (4 custom ML: Human_Anatomy, Human_Identity,
     Instance_Preservation, Multi-View_Consistency) on T4 containers
  2. VBench 2.0 VLM dimensions (13 dims: Composition, Camera_Motion, etc.)
     on CPU containers via Claude Sonnet 4.6 / OpenRouter API
  3. Ads-specific dimensions (8 dims: storyboard_adherence, etc.)
     on CPU containers via Claude Sonnet 4.6 / OpenRouter API

All dimensions launch concurrently — each in its own Modal container.

Usage:
    cd VBench-2.0

    # Run all dimensions on both ads:
    modal run test_ads_modal.py

    # Run on a specific ad:
    modal run test_ads_modal.py --ad dr_doctor
    modal run test_ads_modal.py --ad siyi

    # Run only ads-specific dimensions (no GPU):
    modal run test_ads_modal.py --ads-only

    # Run only VBench dimensions (GPU + VLM API):
    modal run test_ads_modal.py --vbench-only

    # Run only VBench GPU dimensions:
    modal run test_ads_modal.py --gpu-only

    # Run only VBench VLM API dimensions:
    modal run test_ads_modal.py --vlm-only

    # Run specific dimensions:
    modal run test_ads_modal.py --ads-dims "storyboard_adherence,narrative_flow"
    modal run test_ads_modal.py --vbench-dims "Human_Anatomy,Composition"

Requirements:
    - Modal secrets: "aws-credentials", "openrouter-credentials"
    - Ads videos on S3 (auto-uploaded on first run)
"""

import json
import os

import modal

# ── Shared config ──────────────────────────────────────────────────────────

VBENCH_DIR = os.path.dirname(os.path.abspath(__file__))
S3_BUCKET = "video-runner-images"
ADS_S3_PREFIX = "ads-eval"

app = modal.App("ads-eval")

# ── VBench GPU image (reused from test_vbench_modal.py) ───────────────────

MMCV_WHEEL = "mmcv-2.2.0-cp310-cp310-manylinux1_x86_64.whl"
MMCV_WHEEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "wheels", MMCV_WHEEL)
)


def build_vbench_image() -> modal.Image:
    """Heavy image with PyTorch, LLaVA, YOLO-World, etc. for GPU dimensions."""
    image = (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install(
            "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0",
            "git", "wget", "curl", "unzip",
        )
        .pip_install(
            "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
            index_url="https://download.pytorch.org/whl/cu118",
        )
        .add_local_file(MMCV_WHEEL_PATH, f"/tmp/{MMCV_WHEEL}", copy=True)
        .run_commands(
            f"pip install /tmp/{MMCV_WHEEL} --no-cache-dir",
            "pip install setuptools wheel",
        )
        .pip_install("numpy<2", "mmengine==0.10.5", "mmdet==3.0.0")
        .run_commands(
            "pip install 'mmyolo @ git+https://github.com/open-mmlab/mmyolo.git@v0.6.0' "
            "--no-build-isolation --no-cache-dir",
            "pip install 'git+https://github.com/AILab-CVC/YOLO-World.git' "
            "--no-build-isolation --no-cache-dir",
            gpu="T4",
        )
        .pip_install(
            "decord==0.6.0", "opencv-python==4.10.0.84",
            "scipy", "scikit-image", "scikit-learn",
            "einops", "scenedetect", "imageio", "imageio-ffmpeg",
            "gdown", "easydict", "yacs", "kornia",
            "Pillow", "tqdm", "pyyaml", "av",
        )
        .pip_install(
            "transformers==4.51.0", "huggingface-hub", "safetensors",
            "tokenizers>=0.21,<0.22", "accelerate", "sentencepiece",
            "peft", "diffusers",
        )
        .pip_install("git+https://github.com/openai/CLIP.git", extra_options="--no-deps")
        .pip_install("open_clip_torch", extra_options="--no-deps")
        .pip_install("ftfy", "regex")
        .pip_install(
            "llava @ git+https://github.com/LLaVA-VL/LLaVA-NeXT@79ef45a6d8b89b92d7a8525f077c3a3a9894a87d",
            extra_options="--no-deps",
        )
        .pip_install("qwen-vl-utils==0.0.10")
        .pip_install(
            "retinaface-pytorch", "supervision==0.19.0",
            "mediapipe", "ultralytics", "timm==0.4.12", "lvis",
        )
        .pip_install("boto3")
        .run_commands(
            "pip install 'numpy<2' --no-cache-dir",
            "pip install 'transformers==4.51.0' 'tokenizers>=0.21,<0.22' --no-cache-dir",
        )
        .add_local_dir(
            os.path.join(VBENCH_DIR, "vbench2"),
            remote_path="/app/vbench2",
            copy=True,
        )
        .pip_install("modelscope>=1.23", "tensorboard", "datasets")
        .run_commands(
            "pip install -e /app/vbench2/third_party/Instance_detector --no-deps || true"
        )
        .run_commands(
            "sed -i \"s/mmcv_maximum_version = '2.1.0'/mmcv_maximum_version = '2.3.0'/\" "
            "/usr/local/lib/python3.10/site-packages/mmdet/__init__.py",
            "sed -i \"s/mmcv_maximum_version = '2.1.0'/mmcv_maximum_version = '2.3.0'/\" "
            "/usr/local/lib/python3.10/site-packages/mmyolo/__init__.py",
            "find /usr/local/lib/python3.10/site-packages/mmdet -name '__pycache__' -type d -exec rm -rf {} + || true",
            "find /usr/local/lib/python3.10/site-packages/mmyolo -name '__pycache__' -type d -exec rm -rf {} + || true",
            "sed -i 's/self\\.text_feats, None = /self.text_feats, _ = /g' "
            "/usr/local/lib/python3.10/site-packages/yolo_world/models/detectors/yolo_world.py",
            "find /usr/local/lib/python3.10/site-packages/yolo_world -name '__pycache__' -type d -exec rm -rf {} + || true",
            "grep mmcv_maximum_version /usr/local/lib/python3.10/site-packages/mmdet/__init__.py",
            "grep mmcv_maximum_version /usr/local/lib/python3.10/site-packages/mmyolo/__init__.py",
            "sed -i 's/flash_attention_2/sdpa/g' /usr/local/lib/python3.10/site-packages/llava/model/builder.py",
            "find /usr/local/lib/python3.10/site-packages/llava -name '__pycache__' -type d -exec rm -rf {} + || true",
        )
    )
    return image


# ── Ads-only lightweight image (CPU, no torch) ────────────────────────────

ads_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "opencv-python-headless==4.10.0.84",
        "numpy<2",
        "Pillow",
        "requests",
        "scenedetect[opencv]",
        "boto3",
    )
    .add_local_dir(
        os.path.join(VBENCH_DIR, "vbench2"),
        remote_path="/app/vbench2",
        copy=True,
    )
)

vbench_image = build_vbench_image()

# ── VBench GPU dimension config (from test_vbench_modal.py) ───────────────

S3_WEIGHTS_PREFIX = "model-weights/vbench"

S3_WEIGHT_MAP = {
    "Human_Anatomy": [
        ("human_anatomy/yolo_world_v2_xl.pth", "YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"),
        ("human_anatomy/human.pth", "anomaly_detector/human.pth"),
        ("human_anatomy/face.pth", "anomaly_detector/face.pth"),
        ("human_anatomy/hand.pth", "anomaly_detector/hand.pth"),
    ],
    "Human_Identity": [
        ("arcface/resnet18_110.pth", "arcface/resnet18_110.pth"),
    ],
    "Instance_Preservation": [
        ("instance_anomaly_detector/model/adapter_config.json", "instance_anomaly_detector/model/adapter_config.json"),
        ("instance_anomaly_detector/model/adapter_model.safetensors", "instance_anomaly_detector/model/adapter_model.safetensors"),
        ("instance_anomaly_detector/model/additional_config.json", "instance_anomaly_detector/model/additional_config.json"),
        ("instance_anomaly_detector/model/args.json", "instance_anomaly_detector/model/args.json"),
    ],
    "Multi-View_Consistency": [
        ("raft_model/models/raft-things.pth", "raft_model/models/raft-things.pth"),
    ],
}

# 4 GPU-based dims (custom ML models, need T4)
GPU_DIM_TO_MODULE = {
    "Human_Anatomy": "human_anatomy",
}

# VLM dims disabled for ads — Camera_Motion uses irrelevant VBench labels,
# Human_Clothes gives false positives on non-human clips.
# Camera motion now handled by ads_camera_motion in ads dims.
VLM_DIM_TO_MODULE = {}

DIM_TO_MODULE = {**GPU_DIM_TO_MODULE, **VLM_DIM_TO_MODULE}

DIM_TO_INFO_KEY = {}

NO_AUX_DIMS = [
    "Human_Anatomy",
]

ALL_GPU_DIMS = list(GPU_DIM_TO_MODULE.keys())
ALL_VLM_DIMS = list(VLM_DIM_TO_MODULE.keys())
ALL_VBENCH_DIMS = ALL_GPU_DIMS + ALL_VLM_DIMS

ALL_ADS_DIMS = [
    "storyboard_adherence",
    "shot_specific_checks",
    "character_consistency",
    "product_consistency",
    "shot_continuity",
    "narrative_flow",
    "ads_camera_motion",
]


# ── Modal function: VBench GPU dimension ──────────────────────────────────

@app.function(
    image=vbench_image,
    gpu="T4",
    timeout=3600,
    secrets=[
        modal.Secret.from_name("aws-credentials"),
        modal.Secret.from_name("openrouter-credentials"),
    ],
)
def run_vbench_evaluator(
    dimension: str,
    video_s3_key: str,
    ad_id: str,
    eval_config: dict = None,
    s3_bucket: str = None,
) -> dict:
    """Run a VBench 2.0 GPU evaluator on an ads video.

    Uses scene splitting (long video mode) since ads are ~1 min.
    """
    import importlib
    import gc
    import os
    import subprocess
    import sys
    import traceback

    import boto3
    import torch

    sys.path.insert(0, "/app")
    os.chdir("/app")

    CACHE_DIR = os.path.expanduser("~/.cache/vbench2")

    result = {
        "dimension": dimension,
        "ad_id": ad_id,
        "type": "vbench_gpu",
        "status": "error",
        "score": None,
        "error": None,
    }

    try:
        gc.collect()
        torch.cuda.empty_cache()

        # Download video from S3
        bucket = s3_bucket or S3_BUCKET
        video_path = f"/tmp/{ad_id}_video.mp4"
        if not os.path.exists(video_path):
            print(f"Downloading video from s3://{bucket}/{video_s3_key}...")
            s3 = boto3.client("s3")
            s3.download_file(bucket, video_s3_key, video_path)
            print(f"Downloaded: {os.path.getsize(video_path) / 1e6:.1f} MB")

        # Pre-download model weights from S3
        if dimension in S3_WEIGHT_MAP:
            s3 = boto3.client("s3")
            for s3_suffix, cache_subpath in S3_WEIGHT_MAP[dimension]:
                local_path = os.path.join(CACHE_DIR, cache_subpath)
                if not os.path.exists(local_path):
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    s3_key = f"{S3_WEIGHTS_PREFIX}/{s3_suffix}"
                    print(f"  Downloading s3://{S3_BUCKET}/{s3_key} -> {local_path}")
                    s3.download_file(S3_BUCKET, s3_key, local_path)

        if dimension == "Instance_Preservation":
            os.environ["USE_HF"] = "1"

        device = torch.device("cuda")

        import vbench2.hack_registry  # noqa: F401
        from vbench2.utils import init_submodules, save_json

        module_name = DIM_TO_MODULE[dimension]
        dimension_module = importlib.import_module(f"vbench2.{module_name}")
        compute_func = getattr(dimension_module, f"compute_{module_name}")

        print(f"Initializing submodules for {dimension}...")
        submodules_dict = init_submodules([dimension], local=False, read_frame=False)
        submodules_list = submodules_dict[dimension]

        # Split into clips (long video mode — ads are ~1 min)
        # For Human_Anatomy: only create clips for shots with characters
        time_ranges = None
        if dimension == "Human_Anatomy" and eval_config:
            clip_table = eval_config.get("clip_table", [])
            if clip_table:
                shots_by_id = {s["id"]: s for s in eval_config.get("shots", [])}
                time_ranges = []
                for entry in clip_table:
                    shot = shots_by_id.get(entry["shot_id"])
                    if shot and shot.get("has_character", False):
                        time_ranges.append((entry["start_sec"], entry["end_sec"]))
                if time_ranges:
                    print(f"  Filtering Human_Anatomy to {len(time_ranges)} character time ranges")
                else:
                    time_ranges = None

        clip_metas = _split_long_video_into_clips(video_path, time_ranges=time_ranges)
        video_list = [c["path"] for c in clip_metas]
        print(f"  Split video into {len(clip_metas)} clips")

        # Build prompt_dict
        info_path = "/tmp/test_eval_info.json"
        dim_key_lower = DIM_TO_INFO_KEY.get(dimension, dimension.lower())

        if dimension in NO_AUX_DIMS:
            prompt_dict = {
                "prompt_en": "A person walking in a garden",
                "dimension": [dimension],
                "video_list": video_list,
            }
            save_json([prompt_dict], info_path)
        else:
            full_info_path = "/app/vbench2/VBench2_full_info.json"
            with open(full_info_path, "r") as f:
                full_info = json.load(f)

            matching = None
            for entry in full_info:
                if dimension in entry.get("dimension", []):
                    matching = entry
                    break

            if matching is None:
                result["error"] = f"No VBench2_full_info.json entry for {dimension}"
                return result

            prompt_dict = {
                "prompt_en": matching["prompt_en"],
                "dimension": [dimension],
                "video_list": video_list,
            }
            if "auxiliary_info" in matching:
                prompt_dict["auxiliary_info"] = matching["auxiliary_info"]
            if "prompt" in matching:
                prompt_dict["prompt"] = matching["prompt"]
            save_json([prompt_dict], info_path)

        # Run evaluator
        print(f"Running {dimension} evaluator...")
        score, video_results = compute_func(info_path, device, submodules_list)

        result["status"] = "success"
        result["score"] = float(score) if hasattr(score, 'item') else score
        result["num_clips"] = len(video_list)

        # Build per-clip breakdown with time ranges
        clip_time_map = {c["path"]: c for c in clip_metas}
        clips = []
        for idx, vr in enumerate(video_results):
            vp = vr.get("video_path", "")
            meta = clip_time_map.get(vp, {})
            clip_score = vr.get("video_results")
            if hasattr(clip_score, 'item'):
                clip_score = float(clip_score)
            clips.append({
                "clip_idx": idx,
                "start_sec": meta.get("start_sec"),
                "end_sec": meta.get("end_sec"),
                "score": clip_score,
            })
        result["clips"] = clips

        print(f"  Score: {result['score']}")

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["traceback"] = traceback.format_exc()
        print(f"  ERROR: {result['error']}")
        print(traceback.format_exc())

    return result


def _split_long_video_into_clips(
    video_path: str,
    clip_duration: int = 2,
    scene_threshold: float = 27.0,
    time_ranges: list = None,
) -> list:
    """Split a long video into short clips using PySceneDetect + ffmpeg.

    Args:
        time_ranges: Optional list of (start_sec, end_sec) tuples. When provided,
            only creates clips within these ranges (skips scene detection).

    Returns a list of dicts: [{"path": str, "start_sec": float, "end_sec": float}, ...]
    """
    import subprocess
    import json as _json

    clips_dir = "/tmp/long_video_clips"
    os.makedirs(clips_dir, exist_ok=True)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate,duration",
         "-of", "json", video_path],
        capture_output=True, text=True,
    )
    info = _json.loads(probe.stdout)
    duration = float(info["streams"][0]["duration"])
    fr_str = info["streams"][0]["avg_frame_rate"]
    if "/" in fr_str:
        num, den = map(int, fr_str.split("/"))
        fps = num / den if den else 25.0
    else:
        fps = float(fr_str)
    print(f"  Video: {duration:.1f}s, {fps:.1f}fps")

    if time_ranges is not None:
        # Use provided time ranges instead of scene detection
        segments = time_ranges
        print(f"  Using {len(segments)} provided time ranges")
    else:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=scene_threshold))
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        if scene_list:
            segments = [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]
            print(f"  Detected {len(segments)} scenes")
        else:
            segments = [(0.0, duration)]

    clip_metas = []
    clip_idx = 0
    video_stem = os.path.splitext(os.path.basename(video_path))[0]

    for seg_start, seg_end in segments:
        seg_duration = seg_end - seg_start
        num_clips = max(1, int(seg_duration / clip_duration))

        for i in range(num_clips):
            t_start = seg_start + i * clip_duration
            t_end = min(t_start + clip_duration, seg_end)
            if t_end - t_start < 0.5:
                continue

            out_path = os.path.join(clips_dir, f"{video_stem}_clip_{clip_idx:04d}.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path,
                 "-ss", str(t_start), "-t", str(t_end - t_start),
                 "-r", str(fps),
                 "-c:v", "libx264", "-c:a", "aac",
                 "-strict", "experimental",
                 "-loglevel", "error",
                 out_path],
                check=True,
            )
            clip_metas.append({
                "path": out_path,
                "start_sec": round(t_start, 3),
                "end_sec": round(t_end, 3),
            })
            clip_idx += 1

    clip_metas.sort(key=lambda c: c["path"])
    print(f"  Created {len(clip_metas)} clips")
    return clip_metas


# ── Modal function: VBench VLM API dimension (CPU only) ──────────────────

@app.function(
    image=ads_image,
    secrets=[
        modal.Secret.from_name("openrouter-credentials"),
        modal.Secret.from_name("aws-credentials"),
    ],
    timeout=3600,  # 60 min per dimension (some have many API calls)
)
def run_vlm_evaluator(
    dimension: str,
    video_s3_key: str,
    ad_id: str,
    s3_bucket: str = None,
) -> dict:
    """Run a VBench 2.0 VLM evaluator via Claude Sonnet 4.6 / OpenRouter API.

    No GPU needed — extracts frames with OpenCV, sends to API.
    """
    import importlib.util
    import os
    import subprocess
    import sys
    import traceback

    import boto3

    sys.path.insert(0, "/app")
    os.chdir("/app")

    result = {
        "dimension": dimension,
        "ad_id": ad_id,
        "type": "vbench_vlm",
        "status": "error",
        "score": None,
        "error": None,
    }

    try:
        # Download video from S3
        bucket = s3_bucket or S3_BUCKET
        video_path = f"/tmp/{ad_id}_video.mp4"
        if not os.path.exists(video_path):
            print(f"Downloading video from s3://{bucket}/{video_s3_key}...")
            s3 = boto3.client("s3")
            s3.download_file(bucket, video_s3_key, video_path)
            print(f"Downloaded: {os.path.getsize(video_path) / 1e6:.1f} MB")

        # Import vlm_evaluator and openrouter_vlm via importlib (bypass __init__.py)
        for mod_name in ["openrouter_vlm", "vlm_evaluator"]:
            spec = importlib.util.spec_from_file_location(
                f"vbench2.{mod_name}",
                f"/app/vbench2/{mod_name}.py",
                submodule_search_locations=[],
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"vbench2.{mod_name}"] = mod
        for mod_name in ["openrouter_vlm", "vlm_evaluator"]:
            spec = importlib.util.find_spec(f"vbench2.{mod_name}")
            spec.loader.exec_module(sys.modules[f"vbench2.{mod_name}"])

        vlm_eval = sys.modules["vbench2.vlm_evaluator"]
        module_name = VLM_DIM_TO_MODULE[dimension]
        compute_func = getattr(vlm_eval, f"compute_{module_name}")

        # Split into clips (long video mode — ads are ~1 min)
        clip_metas = _split_long_video_into_clips(video_path)
        video_list = [c["path"] for c in clip_metas]
        print(f"  Split video into {len(clip_metas)} clips")

        # Build prompt_dict (same format as VBench expects)
        import json as _json
        info_path = "/tmp/test_eval_info.json"
        dim_key_lower = DIM_TO_INFO_KEY.get(dimension, dimension.lower())

        full_info_path = "/app/vbench2/VBench2_full_info.json"
        with open(full_info_path, "r") as f:
            full_info = _json.load(f)

        matching = None
        for entry in full_info:
            if dimension in entry.get("dimension", []):
                matching = entry
                break

        if matching is None:
            # Dims like Human_Clothes have no auxiliary_info
            prompt_dict = {
                "prompt_en": "A person walking in a garden",
                "dimension": [dimension],
                "video_list": video_list,
            }
        else:
            prompt_dict = {
                "prompt_en": matching["prompt_en"],
                "dimension": [dimension],
                "video_list": video_list,
            }
            if "auxiliary_info" in matching:
                prompt_dict["auxiliary_info"] = matching["auxiliary_info"]
            if "prompt" in matching:
                prompt_dict["prompt"] = matching["prompt"]

        with open(info_path, "w") as f:
            _json.dump([prompt_dict], f)

        # Run evaluator
        print(f"Running VLM {dimension} evaluator...")
        score, video_results = compute_func(info_path, "cpu", {})

        result["status"] = "success"
        result["score"] = float(score) if hasattr(score, 'item') else score
        result["num_clips"] = len(video_list)

        # Build per-clip breakdown with time ranges
        clip_time_map = {c["path"]: c for c in clip_metas}
        clips = []
        for idx, vr in enumerate(video_results):
            vp = vr.get("video_path", "")
            meta = clip_time_map.get(vp, {})
            clip_score = vr.get("video_results")
            if hasattr(clip_score, 'item'):
                clip_score = float(clip_score)
            clips.append({
                "clip_idx": idx,
                "start_sec": meta.get("start_sec"),
                "end_sec": meta.get("end_sec"),
                "score": clip_score,
            })
        result["clips"] = clips

        print(f"  Score: {result['score']}")

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["traceback"] = traceback.format_exc()
        print(f"  ERROR: {result['error']}")
        print(traceback.format_exc())

    return result


# ── Modal function: Ads API dimension (CPU only) ─────────────────────────

@app.function(
    image=ads_image,
    secrets=[
        modal.Secret.from_name("openrouter-credentials"),
        modal.Secret.from_name("aws-credentials"),
    ],
    timeout=1800,  # 30 min per dimension
)
def run_ads_evaluator(
    dimension: str,
    ad_id: str,
    eval_config: dict,
    video_s3_key: str,
    ref_image_keys: dict = None,
    model: str = "qwen/qwen3-vl-235b-a22b-instruct",
    s3_bucket: str = None,
) -> dict:
    """Run an ads-specific evaluator (CPU + OpenRouter API)."""
    import sys
    sys.path.insert(0, "/app")

    import boto3
    import os

    result = {
        "dimension": dimension,
        "ad_id": ad_id,
        "type": "ads_api",
        "status": "error",
        "score": None,
        "error": None,
    }

    try:
        s3 = boto3.client("s3")
        bucket = s3_bucket or S3_BUCKET

        # Download video from S3
        video_path = f"/tmp/{ad_id}_video.mp4"
        if not os.path.exists(video_path):
            print(f"Downloading video from s3://{bucket}/{video_s3_key}...")
            s3.download_file(bucket, video_s3_key, video_path)
            print(f"Downloaded: {os.path.getsize(video_path) / 1e6:.1f} MB")

        # Download ref images from S3 and rewrite paths in config
        ref_images_dir = f"/tmp/{ad_id}_ref_images"
        if ref_image_keys:
            os.makedirs(ref_images_dir, exist_ok=True)
            # Group downloads by section/name
            downloaded = {}  # (section, name) -> [local_path, ...]
            for ref_key, s3_key in ref_image_keys.items():
                local_path = os.path.join(ref_images_dir, os.path.basename(s3_key))
                if not os.path.exists(local_path):
                    print(f"  Downloading ref image: {ref_key}")
                    s3.download_file(S3_BUCKET, s3_key, local_path)
                # Parse key: "section/name/filename"
                parts = ref_key.split("/")
                if len(parts) >= 3:
                    section, name = parts[0], parts[1]
                elif len(parts) == 2:
                    section, name = parts[0], parts[1]
                else:
                    continue
                downloaded.setdefault((section, name), []).append(local_path)
            # Rewrite paths in config
            for (section, name), local_paths in downloaded.items():
                items = eval_config.get("global", {}).get(section, {})
                if name in items and isinstance(items[name], dict):
                    items[name]["ref_images"] = local_paths

        # Override video path in config
        eval_config["video_path"] = video_path

        # Write config to temp file
        config_path = f"/tmp/{ad_id}_eval_config.json"
        with open(config_path, "w") as f:
            json.dump(eval_config, f)

        # Run evaluator — bypass vbench2/__init__.py (which needs torch/gdown/etc)
        import importlib.util
        import sys
        for mod_name in ["openrouter_vlm", "ads_eval"]:
            spec = importlib.util.spec_from_file_location(
                f"vbench2.{mod_name}",
                f"/app/vbench2/{mod_name}.py",
                submodule_search_locations=[],
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"vbench2.{mod_name}"] = mod
        for mod_name in ["openrouter_vlm", "ads_eval"]:
            spec = importlib.util.find_spec(f"vbench2.{mod_name}")
            spec.loader.exec_module(sys.modules[f"vbench2.{mod_name}"])
        AdsEvaluator = sys.modules["vbench2.ads_eval"].AdsEvaluator
        evaluator = AdsEvaluator(config_path, model=model)
        eval_result = evaluator.evaluate_all(dimensions=[dimension])

        dim_result = eval_result["dimensions"].get(dimension, {})
        result["status"] = "success"
        result["score"] = dim_result.get("score", 0)

        # Include extra info
        for key in ["passed", "skipped", "total", "total_checks",
                     "consistent_pairs", "total_pairs",
                     "completeness", "ordering", "coherence",
                     "per_character", "per_product", "matched", "total_checked"]:
            if key in dim_result:
                result[key] = dim_result[key]

        # Strip large responses
        if "details" in dim_result:
            for d in dim_result["details"]:
                for key in ["reason", "response"]:
                    if key in d and isinstance(d[key], str) and len(d[key]) > 200:
                        d[key] = d[key][:200] + "..."
            result["details"] = dim_result["details"]

        print(f"  Score: {result['score']}")

    except Exception as e:
        import traceback
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["traceback"] = traceback.format_exc()
        print(f"  ERROR: {result['error']}")
        print(traceback.format_exc())

    return result


# ── S3 upload helper ──────────────────────────────────────────────────────

def upload_ads_to_s3():
    """Upload ads videos to S3 if not already there. Returns config dict."""
    import boto3
    s3 = boto3.client("s3")

    repo_root = os.path.dirname(VBENCH_DIR)
    ads = {
        "dr_doctor": {
            "video": os.path.join(repo_root, "ads_data", "DR_DOCTOR", "final_video.mp4"),
            "config": os.path.join(repo_root, "ads_data", "DR_DOCTOR", "eval_config.json"),
        },
        "siyi": {
            "video": os.path.join(repo_root, "ads_data", "SIYI", "final_video.mp4"),
            "config": os.path.join(repo_root, "ads_data", "SIYI", "eval_config.json"),
        },
    }

    uploaded = {}
    for ad_id, paths in ads.items():
        video_key = f"{ADS_S3_PREFIX}/{ad_id}/final_video.mp4"

        try:
            s3.head_object(Bucket=S3_BUCKET, Key=video_key)
            print(f"  {ad_id} video already on S3")
        except Exception:
            print(f"  Uploading {ad_id} video to S3...")
            s3.upload_file(paths["video"], S3_BUCKET, video_key)
            print(f"  Done: s3://{S3_BUCKET}/{video_key}")

        with open(paths["config"]) as f:
            config = json.load(f)

        # Upload ref images to S3 and record their S3 keys
        config_dir = os.path.dirname(paths["config"])
        ref_image_keys = {}
        for section in ["products", "characters"]:
            items = config.get("global", {}).get(section, {})
            for name, val in items.items():
                if not isinstance(val, dict):
                    continue
                ref_paths = val.get("ref_images", [])
                if not ref_paths and val.get("ref_image"):
                    ref_paths = [val["ref_image"]]
                for ref_path in ref_paths:
                    local_path = os.path.join(config_dir, ref_path)
                    if os.path.exists(local_path):
                        img_s3_key = f"{ADS_S3_PREFIX}/{ad_id}/ref_images/{os.path.basename(local_path)}"
                        try:
                            s3.head_object(Bucket=S3_BUCKET, Key=img_s3_key)
                        except Exception:
                            print(f"  Uploading ref image {name}: {local_path}")
                            s3.upload_file(local_path, S3_BUCKET, img_s3_key)
                        ref_image_keys[f"{section}/{name}/{os.path.basename(local_path)}"] = img_s3_key

        uploaded[ad_id] = {
            "video_s3_key": video_key,
            "eval_config": config,
            "ref_image_keys": ref_image_keys,
        }

    return uploaded


# ── Main entrypoint ───────────────────────────────────────────────────────

def build_batch_uploaded(batch_spec: str) -> dict:
    """Build uploaded dict from a batch spec string.

    Format: "ad_id:ad_type:s3_uri,ad_id:ad_type:s3_uri,..."
    ad_type is 'dr_doctor' or 'siyi' — determines which eval_config to use.
    s3_uri can be just a key (uses default bucket) or s3://bucket/key.

    If a clip_table.json exists next to final_video.mp4 in S3, it is
    automatically downloaded and merged into the eval_config.
    """
    import boto3
    repo_root = os.path.dirname(VBENCH_DIR)
    config_map = {
        "dr_doctor": os.path.join(repo_root, "ads_data", "DR_DOCTOR", "eval_config.json"),
        "siyi": os.path.join(repo_root, "ads_data", "SIYI", "eval_config.json"),
        # Section-based subtasks
        "siyi_a": os.path.join(repo_root, "ads_data", "SIYI", "sections", "eval_config_a.json"),
        "siyi_b": os.path.join(repo_root, "ads_data", "SIYI", "sections", "eval_config_b.json"),
        "siyi_c": os.path.join(repo_root, "ads_data", "SIYI", "sections", "eval_config_c.json"),
        "siyi_d": os.path.join(repo_root, "ads_data", "SIYI", "sections", "eval_config_d.json"),
        "dr_a": os.path.join(repo_root, "ads_data", "DR_DOCTOR", "sections", "eval_config_a.json"),
        "dr_b": os.path.join(repo_root, "ads_data", "DR_DOCTOR", "sections", "eval_config_b.json"),
        "dr_c": os.path.join(repo_root, "ads_data", "DR_DOCTOR", "sections", "eval_config_c.json"),
        "dr_d": os.path.join(repo_root, "ads_data", "DR_DOCTOR", "sections", "eval_config_d.json"),
        "dr_e": os.path.join(repo_root, "ads_data", "DR_DOCTOR", "sections", "eval_config_e.json"),
    }

    uploaded = {}
    for entry in batch_spec.split(","):
        parts = entry.strip().split(":", 2)
        if len(parts) == 3:
            ad_id, ad_type, s3_uri = parts
        elif len(parts) == 2:
            ad_id, ad_type = parts
            s3_uri = f"{ADS_S3_PREFIX}/{ad_id}/final_video.mp4"
        else:
            print(f"  WARNING: skipping bad batch entry: {entry}")
            continue

        # Parse s3://bucket/key or plain key
        if s3_uri.startswith("s3://"):
            without_prefix = s3_uri[5:]
            bucket, s3_key = without_prefix.split("/", 1)
        else:
            bucket = S3_BUCKET
            s3_key = s3_uri

        config_path = config_map.get(ad_type)
        if not config_path or not os.path.exists(config_path):
            print(f"  WARNING: no eval_config for ad_type '{ad_type}', skipping {ad_id}")
            continue

        with open(config_path) as f:
            config = json.load(f)

        # Try to download clip_table.json from same S3 directory
        s3_dir = s3_key.rsplit("/", 1)[0] if "/" in s3_key else ""
        clip_table_key = f"{s3_dir}/clip_table.json" if s3_dir else "clip_table.json"
        try:
            s3 = boto3.client("s3")
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            s3.download_file(bucket, clip_table_key, tmp.name)
            with open(tmp.name) as f:
                clip_table = json.load(f)
            config["clip_table"] = clip_table
            print(f"  Merged clip_table ({len(clip_table)} entries) from s3://{bucket}/{clip_table_key}")
        except Exception:
            print(f"  No clip_table.json found at s3://{bucket}/{clip_table_key}, using config default")

        # Upload ref images to default S3 bucket (same as upload_ads_to_s3)
        config_dir = os.path.dirname(config_path)
        ref_image_keys = {}
        for section in ["products", "characters"]:
            items = config.get("global", {}).get(section, {})
            for name, val in items.items():
                if not isinstance(val, dict):
                    continue
                # Support both ref_images (list) and ref_image (single)
                ref_paths = val.get("ref_images", [])
                if not ref_paths and val.get("ref_image"):
                    ref_paths = [val["ref_image"]]
                for ref_path in ref_paths:
                    local_path = os.path.join(config_dir, ref_path)
                    if os.path.exists(local_path):
                        img_s3_key = f"{ADS_S3_PREFIX}/{ad_id}/ref_images/{os.path.basename(local_path)}"
                        try:
                            s3.head_object(Bucket=S3_BUCKET, Key=img_s3_key)
                        except Exception:
                            print(f"  Uploading ref image {name}: {local_path}")
                            s3.upload_file(local_path, S3_BUCKET, img_s3_key)
                        ref_image_keys[f"{section}/{name}/{os.path.basename(local_path)}"] = img_s3_key

        uploaded[ad_id] = {
            "video_s3_key": s3_key,
            "eval_config": config,
            "s3_bucket": bucket,
            "ref_image_keys": ref_image_keys,
        }
        print(f"  Batch: {ad_id} (type={ad_type}, s3://{bucket}/{s3_key})")

    return uploaded


@app.local_entrypoint()
def main(
    ad: str = None,
    ads_only: bool = False,
    vbench_only: bool = False,
    gpu_only: bool = False,
    vlm_only: bool = False,
    ads_dims: str = None,
    vbench_dims: str = None,
    model: str = "qwen/qwen3-vl-235b-a22b-instruct",
    batch: str = None,
):
    """Run ads evaluation on Modal — GPU + VLM API + ads API dimensions concurrently."""

    # Determine which dimensions to run
    run_gpu = ALL_GPU_DIMS
    run_vlm = ALL_VLM_DIMS
    run_ads = ALL_ADS_DIMS

    if ads_only:
        run_gpu = []
        run_vlm = []
    elif vbench_only:
        run_ads = []
    elif gpu_only:
        run_vlm = []
        run_ads = []
    elif vlm_only:
        run_gpu = []
        run_ads = []

    if ads_dims:
        run_ads = [d.strip() for d in ads_dims.split(",")]
    if vbench_dims:
        specified = [d.strip() for d in vbench_dims.split(",")]
        run_gpu = [d for d in specified if d in GPU_DIM_TO_MODULE]
        run_vlm = [d for d in specified if d in VLM_DIM_TO_MODULE]

    # Build the set of videos to evaluate
    if batch:
        print("Building batch from spec...")
        uploaded = build_batch_uploaded(batch)
    else:
        print("Checking S3 uploads...")
        uploaded = upload_ads_to_s3()

        if ad:
            if ad not in uploaded:
                print(f"ERROR: Unknown ad '{ad}'. Available: {list(uploaded.keys())}")
                return
            uploaded = {ad: uploaded[ad]}

    total_jobs = len(uploaded) * (len(run_gpu) + len(run_vlm) + len(run_ads))
    print(f"\nLaunching {total_jobs} concurrent evaluations:")
    print(f"  {len(uploaded)} ad(s) x ({len(run_gpu)} GPU + {len(run_vlm)} VLM API + {len(run_ads)} ads API)")

    # Launch everything concurrently
    handles = {}

    for ad_id, info in uploaded.items():
        # Launch VBench GPU dimensions (T4, custom ML models)
        for dim in run_gpu:
            key = f"{ad_id}/gpu/{dim}"
            print(f"  Spawning GPU: {key}")
            handles[key] = run_vbench_evaluator.spawn(
                dimension=dim,
                video_s3_key=info["video_s3_key"],
                ad_id=ad_id,
                eval_config=info.get("eval_config"),
                s3_bucket=info.get("s3_bucket"),
            )

        # Launch VBench VLM dimensions (CPU, Claude Sonnet via OpenRouter)
        for dim in run_vlm:
            key = f"{ad_id}/vlm/{dim}"
            print(f"  Spawning VLM: {key}")
            handles[key] = run_vlm_evaluator.spawn(
                dimension=dim,
                video_s3_key=info["video_s3_key"],
                ad_id=ad_id,
                s3_bucket=info.get("s3_bucket"),
            )

        # Launch ads API dimensions (CPU, Claude Sonnet via OpenRouter)
        for dim in run_ads:
            key = f"{ad_id}/ads/{dim}"
            print(f"  Spawning ADS: {key}")
            handles[key] = run_ads_evaluator.spawn(
                dimension=dim,
                ad_id=ad_id,
                eval_config=info["eval_config"],
                video_s3_key=info["video_s3_key"],
                ref_image_keys=info.get("ref_image_keys"),
                model=model,
                s3_bucket=info.get("s3_bucket"),
            )

    # Collect results
    print(f"\nWaiting for {len(handles)} results...")
    all_results = {}
    for key, handle in handles.items():
        try:
            result = handle.get()
            ad_id = key.split("/")[0]
            if ad_id not in all_results:
                all_results[ad_id] = {"gpu": {}, "vlm": {}, "ads": {}}

            dim_type = key.split("/")[1]  # "gpu", "vlm", or "ads"
            dim_name = key.split("/")[2]
            all_results[ad_id][dim_type][dim_name] = result

            score = result.get("score", "ERROR")
            status = result.get("status", "unknown")
            if status == "success" and isinstance(score, (int, float)):
                print(f"  {key}: {score:.4f}")
            else:
                error = result.get("error", "unknown")[:80]
                print(f"  {key}: {status} — {error}")

        except Exception as e:
            print(f"  {key}: EXCEPTION — {e}")
            ad_id = key.split("/")[0]
            if ad_id not in all_results:
                all_results[ad_id] = {"gpu": {}, "vlm": {}, "ads": {}}
            dim_type = key.split("/")[1]
            dim_name = key.split("/")[2]
            all_results[ad_id][dim_type][dim_name] = {
                "dimension": dim_name, "score": None, "error": str(e)
            }

    # Print summary
    for ad_id, groups in all_results.items():
        print(f"\n{'='*70}")
        print(f"  {ad_id.upper()}")
        print(f"{'='*70}")

        for group_name, group_label in [
            ("gpu", "VBench GPU Dimensions (T4)"),
            ("vlm", "VBench VLM Dimensions (Claude Sonnet API)"),
            ("ads", "Ads-Specific Dimensions (Claude Sonnet API)"),
        ]:
            if groups.get(group_name):
                print(f"\n  {group_label}:")
                print(f"  {'Dimension':<35} {'Score':>8} {'Status':>10}")
                print(f"  {'-'*55}")
                for dim_name, r in sorted(groups[group_name].items()):
                    s = r.get("score")
                    status = r.get("status", "error")
                    if status == "success" and s is not None:
                        print(f"  {dim_name:<35} {s:>8.4f} {'OK':>10}")
                    else:
                        print(f"  {dim_name:<35} {'—':>8} {status:>10}")

        # Overall scores
        all_scores = []
        for group in groups.values():
            if isinstance(group, dict):
                for r in group.values():
                    if r.get("status") == "success" and isinstance(r.get("score"), (int, float)):
                        all_scores.append(r["score"])
        if all_scores:
            print(f"\n  Overall ({len(all_scores)} dimensions): {sum(all_scores)/len(all_scores):.4f}")

    # Save dashboard-ready results — one folder per ad
    from datetime import datetime, timezone
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    results_root = os.path.join(VBENCH_DIR, "output", "eval_results")
    os.makedirs(results_root, exist_ok=True)

    # Dimension metadata for the frontend
    DIM_META = {
        # GPU dims
        "Human_Anatomy": {
            "category": "human_quality",
            "label": "Human Anatomy",
            "description": "Checks for anatomical correctness (face, hands, body proportions) using YOLO-World + ViT anomaly detectors.",
            "aspect": "Are human body parts (face, hands, limbs) anatomically correct and free of distortions?",
        },
        "Human_Identity": {
            "category": "human_quality",
            "label": "Human Identity",
            "description": "Verifies the same person looks consistent across frames using ArcFace face recognition.",
            "aspect": "Does the same person maintain a consistent facial identity throughout?",
        },
        "Instance_Preservation": {
            "category": "consistency",
            "label": "Instance Preservation",
            "description": "Checks that objects/characters maintain their visual identity over time using Qwen2.5-VL.",
            "aspect": "Do key objects and characters preserve their visual identity across frames?",
        },
        "Multi-View_Consistency": {
            "category": "consistency",
            "label": "Multi-View Consistency",
            "description": "Measures 3D consistency using optical flow (RAFT) and dense matching.",
            "aspect": "Is the 3D geometry consistent when the camera moves?",
        },
        # VLM dims
        "Camera_Motion": {
            "category": "cinematography",
            "label": "Camera Motion",
            "description": "Detects camera motion type (pan, tilt, zoom, orbit, static) via VLM analysis.",
            "aspect": "Does the camera motion match the intended movement?",
        },
        "Human_Clothes": {
            "category": "consistency",
            "label": "Human Clothes Consistency",
            "description": "Checks that a person's clothing (color, texture) stays consistent throughout the clip.",
            "aspect": "Do the person's clothes remain visually consistent across frames?",
        },
        # Ads dims
        "storyboard_adherence": {
            "category": "storyboard",
            "label": "Storyboard Adherence",
            "description": "Does each clip match its storyboard shot description?",
            "aspect": "Per-clip match to the storyboard brief.",
        },
        "shot_specific_checks": {
            "category": "storyboard",
            "label": "Shot-Specific Checks",
            "description": "Targeted yes/no questions per shot derived from the storyboard.",
            "aspect": "Does each shot contain the specific elements described in the brief?",
        },
        "character_consistency": {
            "category": "consistency",
            "label": "Character Consistency",
            "description": "Same character looks the same across all their shots.",
            "aspect": "Do characters maintain consistent appearance across shots?",
        },
        "product_consistency": {
            "category": "consistency",
            "label": "Product Consistency",
            "description": "The product looks the same across all product shots.",
            "aspect": "Does the product maintain consistent appearance across shots?",
        },
        "visual_style_unity": {
            "category": "style",
            "label": "Visual Style Unity",
            "description": "Consistent color grading, mood, and visual tone across the ad.",
            "aspect": "Is the visual style (color palette, lighting, mood) consistent?",
        },
        "shot_continuity": {
            "category": "editing",
            "label": "Shot Continuity",
            "description": "Smooth, professional transitions between adjacent shots.",
            "aspect": "Are transitions between shots smooth and free of continuity errors?",
        },
        "narrative_flow": {
            "category": "narrative",
            "label": "Narrative Flow",
            "description": "Does the overall story arc make sense and follow the brief?",
            "aspect": "Is the story coherent, complete, and properly ordered?",
        },
        "lens_atmosphere": {
            "category": "cinematography",
            "label": "Lens & Atmosphere",
            "description": "Cinematographic quality — mood, lighting, and lens effects per shot.",
            "aspect": "Does each shot achieve professional cinematographic quality?",
        },
        "ads_camera_motion": {
            "category": "cinematography",
            "label": "Camera Motion (Storyboard)",
            "description": "Checks if non-static shots match their storyboard-specified camera movement type.",
            "aspect": "Does the camera movement in each shot match the storyboard specification?",
        },
    }

    for ad_id, groups in all_results.items():
        ad_dir = os.path.join(results_root, ad_id)
        os.makedirs(ad_dir, exist_ok=True)

        # Get eval_config for this ad
        ad_config = uploaded.get(ad_id, {}).get("eval_config", {})

        # Build dashboard JSON
        dashboard = {
            "ad_id": ad_id,
            "run_id": run_ts,
            "model": model,
            "video_s3_key": uploaded.get(ad_id, {}).get("video_s3_key", ""),
            "global": ad_config.get("global", {}),
            "overall_score": None,
            "dimensions": [],
        }

        all_scores = []

        for group_name in ["gpu", "vlm", "ads"]:
            group = groups.get(group_name, {})
            for dim_name, r in sorted(group.items()):
                score = r.get("score")
                status = r.get("status", "error")
                meta = DIM_META.get(dim_name, {})

                dim_entry = {
                    "name": dim_name,
                    "label": meta.get("label", dim_name),
                    "category": meta.get("category", "other"),
                    "description": meta.get("description", ""),
                    "aspect": meta.get("aspect", ""),
                    "tier": group_name,  # "gpu", "vlm", or "ads"
                    "status": status,
                    "score": round(score, 4) if isinstance(score, (int, float)) else None,
                    "error": r.get("error"),
                    "num_clips": r.get("num_clips"),
                    "clips": r.get("clips", []),
                    "details": r.get("details", []),
                }

                # Add sub-scores and structured data
                for sub_key in ["completeness", "ordering", "coherence",
                                "passed", "skipped", "total", "total_checks",
                                "consistent_pairs", "total_pairs",
                                "per_character", "per_product",
                                "matched", "total_checked", "total_shots", "matched_shots"]:
                    if sub_key in r:
                        dim_entry[sub_key] = r[sub_key]

                dashboard["dimensions"].append(dim_entry)

                if status == "success" and isinstance(score, (int, float)):
                    all_scores.append(score)

        if all_scores:
            dashboard["overall_score"] = round(sum(all_scores) / len(all_scores), 4)

        # Write dashboard JSON
        output_path = os.path.join(ad_dir, f"eval_{run_ts}.json")
        with open(output_path, "w") as f:
            json.dump(dashboard, f, indent=2, default=str)
        print(f"\n  Dashboard JSON: {output_path}")

        # Also write latest.json as a symlink/copy for easy frontend access
        latest_path = os.path.join(ad_dir, "latest.json")
        with open(latest_path, "w") as f:
            json.dump(dashboard, f, indent=2, default=str)
        print(f"  Latest:         {latest_path}")

    # Also save raw results for debugging
    raw_dir = os.path.join(VBENCH_DIR, "output", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, f"raw_{run_ts}.json")
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Raw results: {raw_path}")
