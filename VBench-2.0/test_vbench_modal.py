"""
VBench 2.0 Evaluator Test on Modal

Runs VBench 2.0 evaluators on a sample video using Modal GPU.
Downloads the video from S3, runs each evaluator, and prints results.

Usage:
    # From VBench-2.0 directory with the venv activated:
    modal run test_vbench_modal.py

    # Run specific dimensions (comma-separated):
    modal run test_vbench_modal.py --dimensions "Human_Anatomy,Human_Identity"

    # Run all 18 dimensions:
    modal run test_vbench_modal.py --all-dims

    # Test with a long video (splits into clips, evaluates each, aggregates):
    modal run test_vbench_modal.py --long-video --dimensions "Human_Anatomy,Composition"
    modal run test_vbench_modal.py --long-video --all-dims
"""
import json
import os

import modal

# ── Modal image definition ──────────────────────────────────────────────
# We build a self-contained image with all VBench 2.0 dependencies.

MMCV_WHEEL = "mmcv-2.2.0-cp310-cp310-manylinux1_x86_64.whl"
MMCV_WHEEL_PATH = os.path.join(
    os.path.dirname(__file__), "wheels", MMCV_WHEEL
)
# Resolve to absolute path
MMCV_WHEEL_PATH = os.path.abspath(MMCV_WHEEL_PATH)

VBENCH_DIR = os.path.dirname(os.path.abspath(__file__))

app = modal.App("vbench2-test")

def build_image() -> modal.Image:
    """Build a Modal image with all VBench 2.0 dependencies."""
    image = (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install(
            "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0",
            "git", "wget", "curl", "unzip",
        )
        # PyTorch with CUDA 11.8
        .pip_install(
            "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
            index_url="https://download.pytorch.org/whl/cu118",
        )
        # mmcv from pre-built wheel
        .add_local_file(MMCV_WHEEL_PATH, f"/tmp/{MMCV_WHEEL}", copy=True)
        .run_commands(
            f"pip install /tmp/{MMCV_WHEEL} --no-cache-dir",
            "pip install setuptools wheel",
        )
        # mmengine + mmdet + mmyolo + YOLO-World
        .pip_install("numpy<2", "mmengine==0.10.5", "mmdet==3.0.0")
        .run_commands(
            "pip install 'mmyolo @ git+https://github.com/open-mmlab/mmyolo.git@v0.6.0' "
            "--no-build-isolation --no-cache-dir",
            "pip install 'git+https://github.com/AILab-CVC/YOLO-World.git' "
            "--no-build-isolation --no-cache-dir",
            gpu="T4",
        )
        # Core ML/vision deps
        .pip_install(
            "decord==0.6.0",
            "opencv-python==4.10.0.84",
            "scipy", "scikit-image", "scikit-learn",
            "einops", "scenedetect", "imageio", "imageio-ffmpeg",
            "gdown", "easydict", "yacs", "kornia",
            "Pillow", "tqdm", "pyyaml", "av",
        )
        # Transformers ecosystem (pin versions for compatibility)
        .pip_install(
            "transformers==4.51.0",
            "huggingface-hub",
            "safetensors",
            "tokenizers>=0.21,<0.22",
            "accelerate",
            "sentencepiece",
            "peft",
            "diffusers",
        )
        # CLIP + open_clip
        .pip_install("git+https://github.com/openai/CLIP.git", extra_options="--no-deps")
        .pip_install("open_clip_torch", extra_options="--no-deps")
        .pip_install("ftfy", "regex")
        # LLaVA-NeXT
        .pip_install(
            "llava @ git+https://github.com/LLaVA-VL/LLaVA-NeXT@79ef45a6d8b89b92d7a8525f077c3a3a9894a87d",
            extra_options="--no-deps",
        )
        .pip_install("qwen-vl-utils==0.0.10")
        # Detection packages
        .pip_install(
            "retinaface-pytorch", "supervision==0.19.0",
            "mediapipe", "ultralytics",
            "timm==0.4.12",
            "lvis",
        )
        # S3 access
        .pip_install("boto3")
        # Force numpy<2 and re-pin transformers at the end (qwen-vl-utils may upgrade it)
        .run_commands(
            "pip install 'numpy<2' --no-cache-dir",
            "pip install 'transformers==4.51.0' 'tokenizers>=0.21,<0.22' --no-cache-dir",
        )
        # Copy VBench 2.0 source code
        .add_local_dir(
            os.path.join(VBENCH_DIR, "vbench2"),
            remote_path="/app/vbench2",
            copy=True,
        )
        # modelscope is needed by Instance_detector's swift/utils/logger.py
        # tensorboard is needed by Instance_detector's swift/utils/tb_utils.py
        # datasets is needed by Instance_detector's swift/utils/torch_utils.py
        .pip_install("modelscope>=1.23", "tensorboard", "datasets")
        # Install Instance_detector (vendored ms-swift for instance_preservation)
        .run_commands(
            "pip install -e /app/vbench2/third_party/Instance_detector --no-deps || true"
        )
        # Patch mmdet + mmyolo version gates LAST (mmcv 2.2.0 vs <2.1.0)
        # Must be after ALL pip installs to avoid being overwritten by reinstalls
        # Also delete __pycache__ .pyc files so Python doesn't use stale bytecode
        .run_commands(
            "sed -i \"s/mmcv_maximum_version = '2.1.0'/mmcv_maximum_version = '2.3.0'/\" "
            "/usr/local/lib/python3.10/site-packages/mmdet/__init__.py",
            "sed -i \"s/mmcv_maximum_version = '2.1.0'/mmcv_maximum_version = '2.3.0'/\" "
            "/usr/local/lib/python3.10/site-packages/mmyolo/__init__.py",
            # Delete bytecode caches so Python re-reads the patched .py files
            "find /usr/local/lib/python3.10/site-packages/mmdet -name '__pycache__' -type d -exec rm -rf {} + || true",
            "find /usr/local/lib/python3.10/site-packages/mmyolo -name '__pycache__' -type d -exec rm -rf {} + || true",
            # Fix YOLO-World SyntaxError: "self.text_feats, None = ..." is
            # invalid in Python 3.10+ (cannot assign to None). Replace with _.
            "sed -i 's/self\\.text_feats, None = /self.text_feats, _ = /g' "
            "/usr/local/lib/python3.10/site-packages/yolo_world/models/detectors/yolo_world.py",
            # Delete yolo_world bytecode cache too
            "find /usr/local/lib/python3.10/site-packages/yolo_world -name '__pycache__' -type d -exec rm -rf {} + || true",
            # Verify the patches applied
            "grep mmcv_maximum_version /usr/local/lib/python3.10/site-packages/mmdet/__init__.py",
            "grep mmcv_maximum_version /usr/local/lib/python3.10/site-packages/mmyolo/__init__.py",
            # Patch pip-installed llava builder.py: default flash_attention_2 -> sdpa
            # (flash_attn package is not installed; sdpa is the PyTorch-native equivalent)
            "sed -i 's/flash_attention_2/sdpa/g' /usr/local/lib/python3.10/site-packages/llava/model/builder.py",
            "find /usr/local/lib/python3.10/site-packages/llava -name '__pycache__' -type d -exec rm -rf {} + || true",
        )
    )
    return image


vbench_image = build_image()

# ── S3 configuration ────────────────────────────────────────────────────

S3_BUCKET = "video-runner-images"
S3_VIDEO_KEY = "test-videos/final_video.mp4"
S3_LONG_VIDEO_KEY = "test-videos/test_long.mp4"
S3_WEIGHTS_PREFIX = "model-weights/vbench"

# S3 weight mappings: (s3_key_suffix, local_cache_subpath)
# These are downloaded BEFORE init_submodules() runs, so it finds them
# already present and skips the Google Drive / Dropbox downloads.
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

# ── Dimension name -> Python module name mapping ───────────────────────

DIM_TO_MODULE = {
    "Human_Anatomy": "human_anatomy",
    "Human_Identity": "human_identity",
    "Human_Clothes": "human_clothes",
    "Diversity": "diversity",
    "Composition": "composition",
    "Dynamic_Spatial_Relationship": "dynamic_spatial_relationship",
    "Dynamic_Attribute": "dynamic_attribute",
    "Motion_Order_Understanding": "motion_order_understanding",
    "Human_Interaction": "human_interaction",
    "Complex_Landscape": "complex_landscape",
    "Complex_Plot": "complex_plot",
    "Camera_Motion": "camera_motion",
    "Motion_Rationality": "motion_rationality",
    "Instance_Preservation": "instance_preservation",
    "Mechanics": "mechanics",
    "Thermotics": "thermotics",
    "Material": "material",
    "Multi-View_Consistency": "multi_view_consistency",
}

# Dimension name -> load_dimension_info dimension string
# (load_dimension_info uses lowercase with underscores, but some have hyphens)
DIM_TO_INFO_KEY = {
    "Multi-View_Consistency": "multi-view_consistency",
}

# ── Evaluator dimensions ────────────────────────────────────────────────

# Dimensions that DON'T need auxiliary_info (work with any video)
NO_AUX_DIMS = [
    "Human_Anatomy", "Human_Identity", "Human_Clothes",
    "Multi-View_Consistency",
]

# All 18 dimensions
ALL_DIMS = [
    "Human_Anatomy", "Human_Identity", "Human_Clothes", "Diversity",
    "Composition", "Dynamic_Spatial_Relationship", "Dynamic_Attribute",
    "Motion_Order_Understanding", "Human_Interaction", "Complex_Landscape",
    "Complex_Plot", "Camera_Motion", "Motion_Rationality",
    "Instance_Preservation", "Mechanics", "Thermotics", "Material",
    "Multi-View_Consistency",
]

# Dimensions that use OpenRouter (Qwen2.5 as judge)
OPENROUTER_DIMS = [
    "Complex_Plot", "Complex_Landscape",
    "Human_Interaction", "Motion_Order_Understanding",
]


@app.function(
    image=vbench_image,
    gpu="A100-80GB",  # 80GB VRAM — needed for long video evaluation with LLaVA-Video-7B
    timeout=7200,  # 120 minutes (long video + Instance_Preservation need extra time)
    secrets=[
        modal.Secret.from_name("aws-credentials"),
        modal.Secret.from_name("openrouter-credentials"),
    ],
)
def run_evaluator(
    dimension: str,
    video_path: str = "/tmp/test_video/final_video.mp4",
    long_video: bool = False,
):
    """Run a single VBench 2.0 evaluator on a video.

    If long_video=True, the video is first split into ~2-second clips using
    PySceneDetect (for scene boundaries) + fixed-duration splitting, then each
    clip is evaluated independently and scores are aggregated.
    """
    import importlib
    import json
    import os
    import subprocess
    import sys
    import traceback

    import boto3
    import torch

    # Add vbench2 to path and set working directory
    # VBench uses relative paths like "vbench2/third_party/..." that
    # assume CWD is the VBench root (parent of vbench2/)
    sys.path.insert(0, "/app")
    os.chdir("/app")

    CACHE_DIR = os.path.expanduser("~/.cache/vbench2")

    result = {
        "dimension": dimension,
        "status": "error",
        "score": None,
        "details": None,
        "error": None,
        "gpu_info": None,
        "long_video": long_video,
    }

    # GPU info
    if torch.cuda.is_available():
        result["gpu_info"] = {
            "name": torch.cuda.get_device_name(0),
            "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        }

    try:
        # ── Free stale GPU memory from warm container reuse ──
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # ── Download video from S3 ──
        os.makedirs("/tmp/test_video", exist_ok=True)
        s3_key = S3_LONG_VIDEO_KEY if long_video else S3_VIDEO_KEY
        if not os.path.exists(video_path):
            print(f"Downloading video from s3://{S3_BUCKET}/{s3_key}...")
            s3 = boto3.client("s3")
            s3.download_file(S3_BUCKET, s3_key, video_path)
        print(f"Video at {video_path}, size: {os.path.getsize(video_path)} bytes")

        # ── Pre-download model weights from S3 ──
        # This places weights where init_submodules() expects them,
        # so it skips the unreliable Google Drive / Dropbox downloads.
        if dimension in S3_WEIGHT_MAP:
            s3 = boto3.client("s3")
            for s3_suffix, cache_subpath in S3_WEIGHT_MAP[dimension]:
                local_path = os.path.join(CACHE_DIR, cache_subpath)
                if not os.path.exists(local_path):
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    s3_key = f"{S3_WEIGHTS_PREFIX}/{s3_suffix}"
                    print(f"  Downloading s3://{S3_BUCKET}/{s3_key} -> {local_path}")
                    s3.download_file(S3_BUCKET, s3_key, local_path)
                    print(f"  Downloaded: {os.path.getsize(local_path)} bytes")

        # For Instance_Preservation: tell swift/modelscope to download
        # the base model from HuggingFace instead of modelscope.cn
        # (modelscope downloads are flaky outside China)
        if dimension == "Instance_Preservation":
            os.environ["USE_HF"] = "1"

        device = torch.device("cuda")

        # Apply hack_registry (patches mmengine Registry to allow
        # duplicate registrations, needed by YOLO-World)
        import vbench2.hack_registry  # noqa: F401 — side effects only

        # Import evaluator and init_submodules
        from vbench2.utils import init_submodules, save_json

        module_name = DIM_TO_MODULE[dimension]

        # Import the compute function
        dimension_module = importlib.import_module(f"vbench2.{module_name}")
        compute_func = getattr(dimension_module, f"compute_{module_name}")

        # Initialize submodules (downloads model weights if needed)
        print(f"Initializing submodules for {dimension}...")
        submodules_dict = init_submodules([dimension], local=False, read_frame=False)
        submodules_list = submodules_dict[dimension]

        # Diversity needs 20 videos with same prompt — skip if we only have 1
        if dimension == "Diversity":
            result["status"] = "skipped"
            result["error"] = "Diversity requires 20 videos per prompt"
            return result

        # ── Long video preprocessing: split into clips ──
        if long_video:
            clip_paths = _split_long_video_into_clips(video_path)
            video_list = clip_paths
            print(f"  Split long video into {len(clip_paths)} clips")
        else:
            video_list = [video_path]

        # ── Build prompt_dict for the evaluator ──
        # Dimensions that need auxiliary_info: use a real entry from VBench2_full_info.json
        # Dimensions without auxiliary_info: use a simple prompt_dict
        info_path = "/tmp/test_eval_info.json"
        dim_key_lower = DIM_TO_INFO_KEY.get(dimension, dimension.lower())

        if dimension in NO_AUX_DIMS:
            # These dimensions don't use auxiliary_info
            prompt_dict = {
                "prompt_en": "A person walking in a garden",
                "dimension": [dimension],
                "video_list": video_list,
            }
            save_json([prompt_dict], info_path)
        else:
            # Load the first matching entry from VBench2_full_info.json
            # and override its video_list to point to our test video
            full_info_path = "/app/vbench2/VBench2_full_info.json"
            with open(full_info_path, "r") as f:
                full_info = json.load(f)

            # Find the first entry for this dimension
            matching = None
            for entry in full_info:
                if dimension in entry.get("dimension", []):
                    matching = entry
                    break

            if matching is None:
                result["status"] = "error"
                result["error"] = f"No VBench2_full_info.json entry for {dimension}"
                return result

            prompt_dict = {
                "prompt_en": matching["prompt_en"],
                "dimension": [dimension],
                "video_list": video_list,
            }
            # Copy auxiliary_info if present
            if "auxiliary_info" in matching:
                prompt_dict["auxiliary_info"] = matching["auxiliary_info"]
            # Some dims also use "prompt" key
            if "prompt" in matching:
                prompt_dict["prompt"] = matching["prompt"]

            save_json([prompt_dict], info_path)
            print(f"  Using prompt: {matching['prompt_en'][:80]}...")
            if "auxiliary_info" in matching:
                ai = matching["auxiliary_info"]
                if isinstance(ai, str):
                    print(f"  auxiliary_info: {ai}")
                elif isinstance(ai, list):
                    print(f"  auxiliary_info: list[{len(ai)}]")
                elif isinstance(ai, dict):
                    print(f"  auxiliary_info: dict keys={list(ai.keys())}")

        # Run the evaluator
        print(f"Running {dimension} evaluator...")
        score, video_results = compute_func(info_path, device, submodules_list)

        if long_video:
            # Aggregate: the evaluator already averages across all clips in video_list,
            # so `score` is the mean over clips. Store clip-level details too.
            result["num_clips"] = len(video_list)

        result["status"] = "success"
        result["score"] = float(score) if hasattr(score, 'item') else score
        result["details"] = video_results
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
) -> list:
    """Split a long video into short clips for evaluation.

    Uses PySceneDetect to find scene boundaries, then splits each scene
    (or the whole video if no scenes detected) into fixed-duration clips
    using ffmpeg. Following vbench2_beta_long's approach.

    Returns a sorted list of clip file paths.
    """
    import subprocess
    import json as _json

    clips_dir = "/tmp/long_video_clips"
    os.makedirs(clips_dir, exist_ok=True)

    # Get video info
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
    print(f"  Long video: {duration:.1f}s, {fps:.1f}fps")

    # Detect scene boundaries with PySceneDetect
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=scene_threshold))
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    if scene_list:
        # Convert scene list to (start_sec, end_sec) pairs
        segments = [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]
        print(f"  Detected {len(segments)} scenes: {[(f'{s:.1f}', f'{e:.1f}') for s, e in segments]}")
    else:
        # No scene transitions — treat the whole video as one segment
        segments = [(0.0, duration)]
        print("  No scene transitions detected, using whole video")

    # Split each scene segment into fixed-duration clips
    clip_paths = []
    clip_idx = 0
    video_stem = os.path.splitext(os.path.basename(video_path))[0]

    for seg_start, seg_end in segments:
        seg_duration = seg_end - seg_start
        num_clips = max(1, int(seg_duration / clip_duration))

        for i in range(num_clips):
            t_start = seg_start + i * clip_duration
            t_end = min(t_start + clip_duration, seg_end)
            # Skip very short clips (< 0.5s)
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
            clip_paths.append(out_path)
            clip_idx += 1

    print(f"  Created {len(clip_paths)} clips in {clips_dir}")
    return sorted(clip_paths)


@app.local_entrypoint()
def main(
    dimensions: str = None,
    all_dims: bool = False,
    long_video: bool = False,
):
    """Run VBench evaluators on Modal.

    Args:
        dimensions: Comma-separated list of dimensions to test.
        all_dims: If True, test all 18 dimensions.
        long_video: If True, use test_long.mp4 and split into clips before evaluation.
    """
    if dimensions:
        dims = [d.strip() for d in dimensions.split(",")]
        # Validate
        for d in dims:
            if d not in ALL_DIMS:
                print(f"Unknown dimension: {d}")
                print(f"Available: {', '.join(ALL_DIMS)}")
                return
    elif all_dims:
        dims = ALL_DIMS
    else:
        dims = NO_AUX_DIMS

    if long_video:
        video_path = "/tmp/test_video/test_long.mp4"
        print(f"Running {len(dims)} evaluators on Modal (LONG VIDEO mode)...\n")
    else:
        video_path = "/tmp/test_video/final_video.mp4"
        print(f"Running {len(dims)} evaluators on Modal...\n")

    # Launch all evaluators concurrently — each runs on its own Modal container
    print(f"Launching all {len(dims)} evaluators concurrently...")
    handles = {}
    for dim in dims:
        handles[dim] = run_evaluator.spawn(dim, video_path=video_path, long_video=long_video)
        print(f"  Spawned: {dim}")

    # Collect results as they complete
    results = {}
    for dim, handle in handles.items():
        result = handle.get()
        results[dim] = result
        status = result["status"]
        if status == "success":
            extra = f" ({result.get('num_clips', 1)} clips)" if long_video else ""
            print(f"  {dim:40s} -> {result['score']:.4f}{extra}")
        elif status == "skipped":
            print(f"  {dim:40s} -> SKIPPED")
        else:
            error = result.get("error", "unknown")[:60]
            print(f"  {dim:40s} -> FAILED: {error}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY" + (" (LONG VIDEO)" if long_video else ""))
    print(f"{'='*60}")
    for dim in dims:
        result = results[dim]
        status = result["status"]
        if status == "success":
            extra = f" ({result.get('num_clips', 1)} clips)" if long_video else ""
            print(f"  {dim:40s} -> {result['score']:.4f}{extra}")
        elif status == "skipped":
            print(f"  {dim:40s} -> SKIPPED")
        else:
            error = result.get("error", "unknown")[:60]
            print(f"  {dim:40s} -> FAILED: {error}")

    # Save full results
    suffix = "_long" if long_video else ""
    output_path = os.path.join(VBENCH_DIR, f"modal_test_results{suffix}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {output_path}")
