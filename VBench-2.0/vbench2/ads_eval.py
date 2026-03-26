"""
Ads Video Evaluation Engine for VBench 2.0.

Evaluates AI-generated advertising videos against storyboard specifications.
Uses VLM via OpenRouter for all VLM evaluation (no local GPU needed).

Evaluation Dimensions:
  1. Storyboard Adherence - does each clip match its storyboard description?
  2. Shot Specific Checks - per-shot targeted yes/no questions
  3. Character Consistency - same character across consecutive shots (per-character sub-scores)
  4. Product Consistency - product matches reference image per shot
  5. Shot Continuity - smooth transitions between adjacent shots
  6. Narrative Flow - coherent story arc matching the brief
  7. Camera Motion - non-static shots match storyboard camera type

Disabled (kept for future use):
  - Visual Style Unity
  - Lens Atmosphere

Usage:
    from vbench2.ads_eval import AdsEvaluator
    evaluator = AdsEvaluator("ads_data/DR_DOCTOR/eval_config.json")
    results = evaluator.evaluate_all()
"""

import json
import os
import subprocess
import tempfile
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Scene detection & storyboard alignment
# ---------------------------------------------------------------------------

def detect_scenes(video_path: str, threshold: float = 27.0) -> List[Tuple[float, float]]:
    """Detect scene boundaries using PySceneDetect.

    Returns list of (start_sec, end_sec) tuples.
    """
    from scenedetect import open_video, SceneManager, ContentDetector

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        # No cuts detected — treat whole video as one scene
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return [(0.0, total / fps)]

    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]


def align_scenes_to_storyboard(
    scenes: List[Tuple[float, float]],
    shots: List[dict],
) -> List[dict]:
    """Align detected scenes to storyboard shots.

    Uses greedy temporal matching: compute expected timestamps from storyboard
    shot durations, then match each storyboard shot to the nearest detected scene.

    Returns shots list with added 'aligned_scene' key containing (start, end) times.
    """
    # Compute expected cumulative timestamps from storyboard
    expected_starts = []
    cumulative = 0.0
    for shot in shots:
        expected_starts.append(cumulative)
        cumulative += shot["duration_sec"]

    total_expected = cumulative
    total_actual = scenes[-1][1] if scenes else 0

    # Scale factor to account for actual vs expected duration difference
    scale = total_actual / total_expected if total_expected > 0 else 1.0

    # For each storyboard shot, find the best matching scene
    aligned_shots = []
    for i, shot in enumerate(shots):
        expected_mid = (expected_starts[i] + expected_starts[i] + shot["duration_sec"]) / 2.0
        expected_mid_scaled = expected_mid * scale

        # Find scene whose midpoint is closest to expected midpoint
        best_scene = None
        best_dist = float("inf")
        for scene_start, scene_end in scenes:
            scene_mid = (scene_start + scene_end) / 2.0
            dist = abs(scene_mid - expected_mid_scaled)
            if dist < best_dist:
                best_dist = dist
                best_scene = (scene_start, scene_end)

        shot_copy = dict(shot)
        shot_copy["aligned_scene"] = best_scene
        aligned_shots.append(shot_copy)

    return aligned_shots


def extract_clip(video_path: str, start: float, end: float, output_path: str):
    """Extract a clip from video using ffmpeg."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y", "-ss", str(start), "-i", video_path,
        "-t", str(duration), "-c:v", "libx264", "-c:a", "aac",
        "-loglevel", "error", output_path,
    ]
    subprocess.run(cmd, check=True)


def extract_frames(video_path: str, n_frames: int = 4, start: float = None, end: float = None) -> List[np.ndarray]:
    """Extract n uniformly-spaced frames from a video (or segment).

    Returns list of numpy arrays (H, W, 3) in RGB.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start is not None and end is not None:
        start_frame = int(start * fps)
        end_frame = min(int(end * fps), total_frames)
    else:
        start_frame = 0
        end_frame = total_frames

    frame_count = end_frame - start_frame
    if frame_count <= 0:
        cap.release()
        return []

    # Uniformly sample frame indices
    if n_frames >= frame_count:
        indices = list(range(start_frame, end_frame))
    else:
        indices = [start_frame + int(i * frame_count / n_frames) for i in range(n_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def extract_transition_frames(
    video_path: str, scene1_end: float, scene2_start: float, n_each: int = 2
) -> List[np.ndarray]:
    """Extract frames around a scene transition (last N of scene1, first N of scene2)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    # Last n_each frames of scene 1
    for i in range(n_each, 0, -1):
        t = max(0, scene1_end - i / fps)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # First n_each frames of scene 2
    for i in range(n_each):
        t = scene2_start + i / fps
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Ads Evaluator
# ---------------------------------------------------------------------------

class AdsEvaluator:
    """Evaluates an ads video against its storyboard config."""

    def __init__(self, config_path: str, api_key: Optional[str] = None,
                 model: str = "anthropic/claude-sonnet-4-6"):
        with open(config_path) as f:
            self.config = json.load(f)

        # Resolve video path relative to config file
        self._config_dir = os.path.dirname(os.path.abspath(config_path))
        video_path = self.config["video_path"]
        if not os.path.isabs(video_path):
            # Try relative to config dir first, then relative to repo root
            candidate = os.path.join(self._config_dir, os.path.basename(video_path))
            if os.path.exists(candidate):
                video_path = candidate
            else:
                # Try from repo root (ads_data/DR_DOCTOR/final_video.mp4)
                repo_root = os.path.dirname(os.path.dirname(self._config_dir))
                video_path = os.path.join(repo_root, video_path)

        self.video_path = video_path
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model
        self.shots = self.config["shots"]
        self.global_info = self.config["global"]
        self._aligned_shots = None

        # Parse clip table (agent-provided shot-to-timestamp mapping)
        raw_clip_table = self.config.get("clip_table", None)
        if raw_clip_table:
            self._clip_table_map = {
                e["shot_id"]: (e["start_sec"], e["end_sec"]) for e in raw_clip_table
            }
        else:
            self._clip_table_map = None

        # Parse characters (backward compat: string or dict)
        self._characters = {}
        self._skip_character_consistency = self.global_info.get("skip_character_consistency", False)
        raw_chars = self.global_info.get("characters", {})
        for name, val in raw_chars.items():
            if isinstance(val, str):
                self._characters[name] = {"description": val, "ref_images": []}
            elif isinstance(val, dict):
                ref_imgs = val.get("ref_images", [])
                if not ref_imgs and val.get("ref_image"):
                    ref_imgs = [val["ref_image"]]
                resolved = [self._resolve_path(p) for p in ref_imgs]
                self._characters[name] = {
                    "description": val.get("description", ""),
                    "ref_images": [r for r in resolved if r],
                }

        # Parse products (new field, optional)
        self._products = {}
        raw_products = self.global_info.get("products", {})
        for name, val in raw_products.items():
            if isinstance(val, dict):
                ref_imgs = val.get("ref_images", [])
                if not ref_imgs and val.get("ref_image"):
                    ref_imgs = [val["ref_image"]]
                resolved = [self._resolve_path(p) for p in ref_imgs]
                self._products[name] = {
                    "description": val.get("description", ""),
                    "ref_images": [r for r in resolved if r],
                }

    def _resolve_path(self, path: Optional[str]) -> Optional[str]:
        """Resolve a relative path against the config directory."""
        if not path:
            return None
        if os.path.isabs(path):
            return path
        resolved = os.path.join(self._config_dir, path)
        if os.path.exists(resolved):
            return resolved
        # Try from repo root
        repo_root = os.path.dirname(os.path.dirname(self._config_dir))
        resolved2 = os.path.join(repo_root, path)
        if os.path.exists(resolved2):
            return resolved2
        return resolved  # return config-relative even if not found

    def _get_aligned_shots(self) -> List[dict]:
        """Get shots with aligned_scene timestamps (cached).

        If clip_table is provided in config, uses it directly.
        Otherwise falls back to PySceneDetect alignment.
        """
        if self._aligned_shots is None:
            if self._clip_table_map:
                # Use agent-provided clip table
                print(f"  Using clip table ({len(self._clip_table_map)} entries) "
                      f"for {len(self.shots)} storyboard shots...")
                self._aligned_shots = []
                for shot in self.shots:
                    shot_copy = dict(shot)
                    if shot["id"] in self._clip_table_map:
                        shot_copy["aligned_scene"] = self._clip_table_map[shot["id"]]
                        shot_copy["in_clip_table"] = True
                    else:
                        shot_copy["aligned_scene"] = None
                        shot_copy["in_clip_table"] = False
                    self._aligned_shots.append(shot_copy)
            else:
                # Fallback: PySceneDetect alignment
                print(f"  Detecting scenes in {self.video_path}...")
                scenes = detect_scenes(self.video_path)
                print(f"  Found {len(scenes)} scenes, aligning to {len(self.shots)} storyboard shots...")
                self._aligned_shots = align_scenes_to_storyboard(scenes, self.shots)
                for shot in self._aligned_shots:
                    shot["in_clip_table"] = True
        return self._aligned_shots

    def _get_present_shots(self) -> List[dict]:
        """Return only shots present in clip table (with valid aligned_scene)."""
        return [s for s in self._get_aligned_shots()
                if s.get("in_clip_table", True) and s.get("aligned_scene") is not None]

    def _query(self, prompt: str, frames: List[np.ndarray] = None, **kwargs) -> str:
        """Query the VLM."""
        from .openrouter_vlm import query_vlm
        return query_vlm(
            prompt=prompt,
            frames=frames,
            api_key=self.api_key,
            model=self.model,
            **kwargs,
        )

    def _query_yes_no(self, prompt: str, frames: List[np.ndarray] = None, **kwargs) -> Tuple[bool, str]:
        """Query VLM with yes/no question."""
        from .openrouter_vlm import query_vlm_yes_no
        return query_vlm_yes_no(
            prompt=prompt,
            frames=frames,
            api_key=self.api_key,
            model=self.model,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Dimension 1: Storyboard Adherence
    # ------------------------------------------------------------------
    def compute_storyboard_adherence(self) -> Dict:
        """Does each clip match its storyboard shot description?

        Scoring: correct match=1.0, honest skip (not in clip table)=0.5, wrong match=0.0.
        """
        all_shots = self._get_aligned_shots()
        results = []

        for shot in all_shots:
            # Shot not in clip table — honest skip, score 0.5
            if not shot.get("in_clip_table", True) or shot.get("aligned_scene") is None:
                results.append({
                    "shot_id": shot["id"], "score": 0.5,
                    "status": "skipped", "reason": "Not in clip table",
                })
                print(f"    Shot {shot['id']}: SKIPPED (0.5)")
                continue

            scene = shot["aligned_scene"]
            frames = extract_frames(self.video_path, n_frames=4, start=scene[0], end=scene[1])
            if not frames:
                results.append({"shot_id": shot["id"], "start_sec": round(scene[0], 3),
                                "end_sec": round(scene[1], 3), "score": 0,
                                "status": "fail", "reason": "No frames extracted"})
                continue

            prompt = (
                f"This clip is from an advertisement. The storyboard says this shot should show:\n"
                f"\"{shot['description_en']}\"\n\n"
                f"Does this clip match the storyboard description? "
                f"Answer YES or NO first, then give a brief reason."
            )
            is_yes, reason = self._query_yes_no(prompt, frames)
            results.append({
                "shot_id": shot["id"],
                "start_sec": round(scene[0], 3),
                "end_sec": round(scene[1], 3),
                "score": 1.0 if is_yes else 0.0,
                "status": "pass" if is_yes else "fail",
                "reason": reason,
            })
            print(f"    Shot {shot['id']}: {'PASS' if is_yes else 'FAIL'}")

        score = sum(r["score"] for r in results) / len(results) if results else 0
        return {"dimension": "Ads_Storyboard_Adherence", "score": score, "details": results}

    # ------------------------------------------------------------------
    # Dimension 2: Shot Specific Checks
    # ------------------------------------------------------------------
    def compute_shot_specific_checks(self) -> Dict:
        """Per-shot targeted yes/no questions from specific_checks.

        Scoring: pass=1.0, fail=0.0, skipped (not in clip table)=0.5 per check.
        """
        all_shots = self._get_aligned_shots()
        results = []
        total_score = 0.0
        total_checks = 0

        for shot in all_shots:
            checks = shot.get("specific_checks", [])
            if not checks:
                continue

            # Shot not in clip table — honest skip, 0.5 per check
            if not shot.get("in_clip_table", True) or shot.get("aligned_scene") is None:
                for check in checks:
                    results.append({
                        "shot_id": shot["id"], "check": check,
                        "passed": None, "status": "skipped",
                        "score": 0.5, "reason": "Not in clip table",
                    })
                    total_score += 0.5
                    total_checks += 1
                print(f"    Shot {shot['id']}: SKIPPED ({len(checks)} checks x 0.5)")
                continue

            scene = shot["aligned_scene"]
            frames = extract_frames(self.video_path, n_frames=4, start=scene[0], end=scene[1])
            if not frames:
                for check in checks:
                    results.append({"shot_id": shot["id"], "start_sec": round(scene[0], 3),
                                    "end_sec": round(scene[1], 3), "check": check,
                                    "passed": False, "status": "fail",
                                    "score": 0.0, "reason": "No frames"})
                    total_checks += 1
                continue

            for check in checks:
                is_yes, reason = self._query_yes_no(check, frames)
                check_score = 1.0 if is_yes else 0.0
                results.append({
                    "shot_id": shot["id"],
                    "start_sec": round(scene[0], 3),
                    "end_sec": round(scene[1], 3),
                    "check": check,
                    "passed": is_yes,
                    "status": "pass" if is_yes else "fail",
                    "score": check_score,
                    "reason": reason,
                })
                total_score += check_score
                total_checks += 1
                print(f"    Shot {shot['id']} check: {'PASS' if is_yes else 'FAIL'} — {check[:60]}...")

        score = total_score / total_checks if total_checks > 0 else 0
        passed = sum(1 for r in results if r.get("status") == "pass")
        skipped = sum(1 for r in results if r.get("status") == "skipped")
        return {"dimension": "Ads_Shot_Specific_Checks", "score": score,
                "passed": passed, "skipped": skipped, "total": total_checks,
                "details": results}

    # ------------------------------------------------------------------
    # Dimension 3: Character Consistency
    # ------------------------------------------------------------------
    def compute_character_consistency(self) -> Dict:
        """Character matches reference images across shots. Per-character sub-scores."""
        from PIL import Image

        if self._skip_character_consistency:
            print("    Character consistency: skipped (no fixed characters)")
            return {"dimension": "Ads_Character_Consistency", "score": 1.0,
                    "details": [], "note": "Skipped — no fixed characters for this ad"}

        present = self._get_present_shots()
        all_results = []
        per_character = {}
        total_passed = 0
        total_failed = 0

        for char_name, char_info in self._characters.items():
            char_desc = char_info["description"]
            ref_images = char_info.get("ref_images", [])
            char_shots = [s for s in present if char_name in s.get("characters", [])]
            if not char_shots:
                continue

            # Load ref image frames
            ref_frames = []
            for ref_path in ref_images:
                if ref_path and os.path.exists(ref_path):
                    img = Image.open(ref_path).convert("RGB")
                    ref_frames.append(np.array(img))

            char_results = []
            char_passed = 0
            char_failed = 0

            if ref_frames:
                # Reference-based: compare each shot against ref images
                for shot in char_shots:
                    frames = extract_frames(self.video_path, n_frames=3,
                                            start=shot["aligned_scene"][0], end=shot["aligned_scene"][1])
                    s_start = round(shot["aligned_scene"][0], 1)
                    s_end = round(shot["aligned_scene"][1], 1)
                    if not frames:
                        char_results.append({
                            "character": char_name, "shot_id": shot["id"],
                            "start_sec": s_start, "end_sec": s_end,
                            "score": 0.0, "status": "fail",
                            "consistent": False, "reason": "No frames"})
                        continue

                    all_frames = ref_frames + frames
                    prompt = (
                        f"The first {len(ref_frames)} image(s) are reference photos of "
                        f"'{char_name}' ({char_desc}).\n"
                        f"The remaining {len(frames)} frames are from shot {shot['id']} "
                        f"of an advertisement.\n\n"
                        f"Does this shot show ANY of the people from the reference photos?\n"
                        f"- The shot may show fewer people than the reference (e.g. only one of a duo). "
                        f"That is fine — just check whether any visible person matches.\n"
                        f"- Focus on face shape, hairstyle, hair color, body type, approximate age. "
                        f"Ignore changes in pose, expression, or exact clothing.\n"
                        f"- If faces are not visible (back to camera, silhouette, heavy backlighting), "
                        f"answer SKIP.\n\n"
                        f"Answer YES, NO, or SKIP first, then explain briefly."
                    )
                    raw_reason = self._query(prompt, all_frames)
                    first_word = raw_reason.strip().split()[0].upper().rstrip(".,!:;")
                    if first_word == "SKIP":
                        char_results.append({
                            "character": char_name, "shot_id": shot["id"],
                            "start_sec": s_start, "end_sec": s_end,
                            "score": -1.0, "status": "skip",
                            "consistent": None, "reason": raw_reason,
                        })
                        print(f"    {char_name} shot {shot['id']}: SKIP")
                        continue
                    is_yes = first_word == "YES"
                    if is_yes:
                        char_passed += 1
                    else:
                        char_failed += 1
                    char_results.append({
                        "character": char_name, "shot_id": shot["id"],
                        "start_sec": s_start, "end_sec": s_end,
                        "score": 1.0 if is_yes else 0.0,
                        "status": "pass" if is_yes else "fail",
                        "consistent": is_yes, "reason": raw_reason,
                    })
                    print(f"    {char_name} shot {shot['id']}: "
                          f"{'MATCH' if is_yes else 'MISMATCH'}")
            else:
                # Fallback: pairwise comparison (no ref images)
                for i in range(len(char_shots) - 1):
                    s1 = char_shots[i]
                    s2 = char_shots[i + 1]
                    frames1 = extract_frames(self.video_path, n_frames=3,
                                             start=s1["aligned_scene"][0], end=s1["aligned_scene"][1])
                    frames2 = extract_frames(self.video_path, n_frames=3,
                                             start=s2["aligned_scene"][0], end=s2["aligned_scene"][1])
                    s1_start = round(s1["aligned_scene"][0], 1)
                    s1_end = round(s1["aligned_scene"][1], 1)
                    s2_start = round(s2["aligned_scene"][0], 1)
                    s2_end = round(s2["aligned_scene"][1], 1)
                    if not frames1 or not frames2:
                        char_results.append({
                            "character": char_name, "shots": [s1["id"], s2["id"]],
                            "start_sec": s1_start, "end_sec": s2_end,
                            "shot_times": [{"start_sec": s1_start, "end_sec": s1_end},
                                           {"start_sec": s2_start, "end_sec": s2_end}],
                            "score": 0.0, "status": "fail",
                            "consistent": False, "reason": "No frames"})
                        continue

                    combined = frames1 + frames2
                    prompt = (
                        f"These frames are from two different shots of the same advertisement.\n"
                        f"The first {len(frames1)} frames are from shot {s1['id']}, "
                        f"the last {len(frames2)} frames are from shot {s2['id']}.\n"
                        f"The character '{char_name}' ({char_desc}) should appear in both.\n\n"
                        f"Do these two shots show the same person as '{char_name}'? "
                        f"Focus on face shape, hairstyle, hair color, body type, approximate age. "
                        f"Ignore changes in pose, expression, or exact clothing.\n"
                        f"Answer YES or NO first, then explain briefly."
                    )
                    is_yes, reason = self._query_yes_no(prompt, combined)
                    if is_yes:
                        char_passed += 1
                    else:
                        char_failed += 1
                    char_results.append({
                        "character": char_name, "shots": [s1["id"], s2["id"]],
                        "start_sec": s1_start, "end_sec": s2_end,
                        "shot_times": [{"start_sec": s1_start, "end_sec": s1_end},
                                       {"start_sec": s2_start, "end_sec": s2_end}],
                        "score": 1.0 if is_yes else 0.0,
                        "status": "pass" if is_yes else "fail",
                        "consistent": is_yes, "reason": reason,
                    })
                    print(f"    {char_name} shots {s1['id']}->{s2['id']}: "
                          f"{'CONSISTENT' if is_yes else 'INCONSISTENT'}")

            n_judged = char_passed + char_failed
            per_character[char_name] = {
                "score": char_passed / n_judged if n_judged > 0 else 1.0,
                "total_checks": len(char_results),
                "passed": char_passed,
                "failed": char_failed,
                "skipped": len(char_results) - n_judged,
                "details": char_results,
            }
            total_passed += char_passed
            total_failed += char_failed
            all_results.extend(char_results)

        total_judged = total_passed + total_failed
        score = total_passed / total_judged if total_judged > 0 else 1.0
        return {"dimension": "Ads_Character_Consistency", "score": score,
                "passed": total_passed, "failed": total_failed,
                "total_checks": len(all_results),
                "skipped": len(all_results) - total_judged,
                "per_character": per_character, "details": all_results}

    # ------------------------------------------------------------------
    # Dimension 4: Product Consistency
    # ------------------------------------------------------------------
    def compute_product_consistency(self) -> Dict:
        """Product matches reference image in each product shot."""
        from PIL import Image
        present = self._get_present_shots()
        product_shots = [s for s in present if s.get("has_product")]

        if not product_shots:
            return {"dimension": "Ads_Product_Consistency", "score": 1.0,
                    "details": [], "note": "No product shots"}

        # Determine products to check
        products = self._products
        if not products:
            # Legacy fallback: single product from product_description
            products = {"product": {
                "description": self.global_info.get("product_description", ""),
                "ref_images": [],
            }}

        all_results = []
        per_product = {}

        for prod_name, prod_info in products.items():
            ref_image_paths = prod_info.get("ref_images", [])
            prod_desc = prod_info["description"]

            # Load ref images as numpy arrays
            ref_frames = []
            for ref_path in ref_image_paths:
                if ref_path and os.path.exists(ref_path):
                    img = Image.open(ref_path).convert("RGB")
                    ref_frames.append(np.array(img))

            prod_results = []
            for shot in product_shots:
                scene = shot["aligned_scene"]
                frames = extract_frames(self.video_path, n_frames=3,
                                        start=scene[0], end=scene[1])
                if not frames:
                    prod_results.append({
                        "shot_id": shot["id"], "product": prod_name,
                        "start_sec": round(scene[0], 3), "end_sec": round(scene[1], 3),
                        "score": 0, "reason": "No frames"})
                    continue

                if ref_frames:
                    # Prepend ref images so VLM sees them first
                    all_frames = ref_frames + frames
                    prompt = (
                        f"The first {len(ref_frames)} image(s) are reference photos of the product "
                        f"'{prod_name}' ({prod_desc}).\n"
                        f"The remaining {len(frames)} frames are from shot {shot['id']} "
                        f"of an advertisement.\n\n"
                        f"Does this shot show a product that matches ANY of the reference images? "
                        f"The product line has multiple variants (different bottle sizes/types). "
                        f"Focus on brand identity, bottle style, color scheme, label design.\n"
                        f"Answer YES or NO first, then explain briefly."
                    )
                else:
                    all_frames = frames
                    prompt = (
                        f"These frames are from shot {shot['id']} of an advertisement.\n"
                        f"The product should be: {prod_desc}\n\n"
                        f"Does this shot show a product matching that description? "
                        f"Answer YES or NO first, then explain briefly."
                    )

                is_yes, reason = self._query_yes_no(prompt, all_frames)
                score = 1.0 if is_yes else 0.0
                prod_results.append({
                    "shot_id": shot["id"], "product": prod_name,
                    "start_sec": round(scene[0], 3), "end_sec": round(scene[1], 3),
                    "score": score, "reason": reason,
                })
                print(f"    Product '{prod_name}' shot {shot['id']}: "
                      f"{'MATCH' if is_yes else 'MISMATCH'}")

            prod_score = (sum(r["score"] for r in prod_results) / len(prod_results)
                          if prod_results else 1.0)
            broken = [r["shot_id"] for r in prod_results if r["score"] == 0]
            per_product[prod_name] = {
                "score": prod_score,
                "total_shots": len(prod_results),
                "broken_shots": broken,
                "details": prod_results,
            }
            all_results.extend(prod_results)

        overall = (sum(r["score"] for r in all_results) / len(all_results)
                   if all_results else 1.0)
        return {"dimension": "Ads_Product_Consistency", "score": overall,
                "per_product": per_product, "details": all_results}

    # ------------------------------------------------------------------
    # Dimension 5: Visual Style Unity
    # ------------------------------------------------------------------
    def compute_visual_style_unity(self) -> Dict:
        """Consistent visual style (color, mood, tone) across the ad."""
        aligned = self._get_aligned_shots()
        style_desc = self.global_info.get("style", "")
        palette = self.global_info.get("color_palette", [])

        # Sample up to 10 random pairs from different parts of the video
        n_shots = len(aligned)
        pairs = []
        if n_shots >= 4:
            # Sample pairs from different quarters
            quarter = n_shots // 4
            for _ in range(min(10, n_shots)):
                i = random.randint(0, n_shots // 2 - 1)
                j = random.randint(n_shots // 2, n_shots - 1)
                pairs.append((i, j))
            # Deduplicate
            pairs = list(set(pairs))[:10]
        else:
            pairs = [(i, j) for i in range(n_shots) for j in range(i + 1, n_shots)]

        results = []
        total_consistent = 0

        for i, j in pairs:
            s1 = aligned[i]
            s2 = aligned[j]
            frames1 = extract_frames(self.video_path, n_frames=2,
                                     start=s1["aligned_scene"][0], end=s1["aligned_scene"][1])
            frames2 = extract_frames(self.video_path, n_frames=2,
                                     start=s2["aligned_scene"][0], end=s2["aligned_scene"][1])

            if not frames1 or not frames2:
                results.append({"shots": [s1["id"], s2["id"]],
                                "shot_times": [{"start_sec": round(s1["aligned_scene"][0], 3), "end_sec": round(s1["aligned_scene"][1], 3)},
                                               {"start_sec": round(s2["aligned_scene"][0], 3), "end_sec": round(s2["aligned_scene"][1], 3)}],
                                "consistent": False, "reason": "No frames"})
                continue

            combined = frames1 + frames2
            prompt = (
                f"These frames are from two different parts of the same advertisement.\n"
                f"First {len(frames1)} frames: shot {s1['id']} ({s1['description_en'][:50]}...)\n"
                f"Last {len(frames2)} frames: shot {s2['id']} ({s2['description_en'][:50]}...)\n\n"
                f"The ad's intended visual style is: {style_desc}\n"
                f"Intended color palette: {', '.join(palette)}\n\n"
                f"Do these two clips maintain a consistent visual style "
                f"(color grading, lighting mood, overall tone)? "
                f"Some variation between shot types (close-up vs wide, indoor vs outdoor) is natural, "
                f"but the overall palette and mood should feel unified.\n"
                f"Answer YES or NO first, then explain briefly."
            )
            is_yes, reason = self._query_yes_no(prompt, combined)
            if is_yes:
                total_consistent += 1
            results.append({
                "shots": [s1["id"], s2["id"]],
                "shot_times": [{"start_sec": round(s1["aligned_scene"][0], 3), "end_sec": round(s1["aligned_scene"][1], 3)},
                                {"start_sec": round(s2["aligned_scene"][0], 3), "end_sec": round(s2["aligned_scene"][1], 3)}],
                "consistent": is_yes,
                "reason": reason,
            })
            print(f"    Style unity shots {s1['id']} vs {s2['id']}: "
                  f"{'CONSISTENT' if is_yes else 'INCONSISTENT'}")

        score = total_consistent / len(pairs) if pairs else 1.0
        return {"dimension": "Ads_Visual_Style_Unity", "score": score,
                "consistent_pairs": total_consistent, "total_pairs": len(pairs), "details": results}

    # ------------------------------------------------------------------
    # Dimension 6: Shot Continuity
    # ------------------------------------------------------------------
    def compute_shot_continuity(self) -> Dict:
        """Smooth transitions between adjacent shots."""
        present = self._get_present_shots()
        # Sort by start time to ensure correct transition order
        present = sorted(present, key=lambda s: s["aligned_scene"][0])
        results = []
        total_score = 0.0

        for i in range(len(present) - 1):
            s1 = present[i]
            s2 = present[i + 1]
            scene1_end = s1["aligned_scene"][1]
            scene2_start = s2["aligned_scene"][0]

            frames = extract_transition_frames(self.video_path, scene1_end, scene2_start, n_each=2)
            if len(frames) < 2:
                results.append({"transition": [s1["id"], s2["id"]],
                                "transition_sec": round(scene1_end, 3),
                                "shot_times": [{"start_sec": round(s1["aligned_scene"][0], 3), "end_sec": round(s1["aligned_scene"][1], 3)},
                                               {"start_sec": round(s2["aligned_scene"][0], 3), "end_sec": round(s2["aligned_scene"][1], 3)}],
                                "score": 0.5, "reason": "Insufficient frames"})
                total_score += 0.5
                continue

            prompt = (
                f"These {len(frames)} frames show the transition between two adjacent shots "
                f"in an advertisement.\n"
                f"The first frames are from the end of shot {s1['id']} "
                f"({s1['description_en'][:50]}...).\n"
                f"The last frames are from the start of shot {s2['id']} "
                f"({s2['description_en'][:50]}...).\n\n"
                f"Rate the transition quality:\n"
                f"- SMOOTH: Professional, intentional transition (clean cut, dissolve, or motivated edit)\n"
                f"- MINOR_ISSUES: Acceptable but slightly jarring (small continuity errors)\n"
                f"- MAJOR_BREAK: Severe continuity error (impossible jumps, visual glitches)\n\n"
                f"Answer with exactly one of: SMOOTH, MINOR_ISSUES, or MAJOR_BREAK. "
                f"Then explain briefly."
            )
            response = self._query(prompt, frames)
            upper = response.strip().upper()
            if "SMOOTH" in upper.split("\n")[0]:
                t_score = 1.0
            elif "MINOR" in upper.split("\n")[0]:
                t_score = 0.5
            else:
                t_score = 0.0

            total_score += t_score
            results.append({
                "transition": [s1["id"], s2["id"]],
                "transition_sec": round(scene1_end, 3),
                "shot_times": [{"start_sec": round(s1["aligned_scene"][0], 3), "end_sec": round(s1["aligned_scene"][1], 3)},
                               {"start_sec": round(s2["aligned_scene"][0], 3), "end_sec": round(s2["aligned_scene"][1], 3)}],
                "score": t_score,
                "reason": response,
            })
            label = "SMOOTH" if t_score == 1.0 else ("MINOR" if t_score == 0.5 else "MAJOR_BREAK")
            print(f"    Transition {s1['id']}->{s2['id']}: {label}")

        n = len(present) - 1
        score = total_score / n if n > 0 else 1.0
        return {"dimension": "Ads_Shot_Continuity", "score": score, "details": results}

    # ------------------------------------------------------------------
    # Dimension 7: Narrative Flow
    # ------------------------------------------------------------------
    def compute_narrative_flow(self) -> Dict:
        """Does the overall video tell a coherent story matching the brief?"""
        # Sample frames broadly across the full video
        frames = extract_frames(self.video_path, n_frames=16)
        if not frames:
            return {"dimension": "Ads_Narrative_Flow", "score": 0, "reason": "No frames extracted"}

        narrative = self.global_info.get("narrative_summary", "")
        beats = self.global_info.get("story_beats", [])
        if beats:
            story_beats = "\n".join(f"- {b}" for b in beats)
        else:
            # Fallback: use per-shot descriptions
            story_beats = "\n".join(f"- {s['description_en']}" for s in self.shots)

        prompt = (
            f"These {len(frames)} frames are uniformly sampled from a "
            f"{self.global_info.get('total_duration_sec', 60)}-second advertisement.\n\n"
            f"The intended narrative is:\n{narrative}\n\n"
            f"The story should progress through these beats:\n{story_beats}\n\n"
            f"Based on the frames, evaluate the narrative quality on three criteria:\n\n"
            f"1. COMPLETENESS (0.0-1.0): What fraction of the major story beats "
            f"appear to be present in the video?\n"
            f"2. ORDERING (0.0-1.0): Are the story beats in approximately the correct sequence? "
            f"1.0=correct order, 0.5=mostly correct, 0.0=jumbled\n"
            f"3. COHERENCE (0.0-1.0): Does the narrative flow naturally as a story? "
            f"1.0=smooth and clear, 0.5=somewhat disjointed, 0.0=incoherent\n\n"
            f"Answer in this exact format:\n"
            f"COMPLETENESS: X.X\n"
            f"ORDERING: X.X\n"
            f"COHERENCE: X.X\n"
            f"EXPLANATION: (brief reasoning)"
        )
        response = self._query(prompt, frames, max_tokens=1500)

        # Parse scores from response
        scores = {}
        for line in response.strip().split("\n"):
            line = line.strip()
            for key in ["COMPLETENESS", "ORDERING", "COHERENCE"]:
                if line.upper().startswith(key):
                    try:
                        val = float(line.split(":")[-1].strip().split()[0])
                        scores[key.lower()] = max(0.0, min(1.0, val))
                    except (ValueError, IndexError):
                        scores[key.lower()] = 0.5

        completeness = scores.get("completeness", 0.5)
        ordering = scores.get("ordering", 0.5)
        coherence = scores.get("coherence", 0.5)

        # Weighted average
        score = 0.4 * completeness + 0.3 * ordering + 0.3 * coherence

        print(f"    Narrative: completeness={completeness:.2f}, "
              f"ordering={ordering:.2f}, coherence={coherence:.2f}, total={score:.2f}")

        return {
            "dimension": "Ads_Narrative_Flow",
            "score": score,
            "completeness": completeness,
            "ordering": ordering,
            "coherence": coherence,
            "response": response,
        }

    # ------------------------------------------------------------------
    # Dimension 8: Lens & Atmosphere
    # ------------------------------------------------------------------
    def compute_lens_atmosphere(self) -> Dict:
        """Cinematographic quality — mood, lighting, lens effects per clip."""
        aligned = self._get_aligned_shots()
        style_desc = self.global_info.get("style", "")
        results = []
        total_score = 0.0
        evaluated = 0

        for shot in aligned:
            scene = shot["aligned_scene"]
            frames = extract_frames(self.video_path, n_frames=3, start=scene[0], end=scene[1])
            if not frames:
                results.append({
                    "shot_id": shot["id"],
                    "start_sec": round(scene[0], 3),
                    "end_sec": round(scene[1], 3),
                    "score": 0,
                    "response": "No frames extracted",
                })
                continue

            prompt = (
                f"These frames are from shot {shot['id']} of an advertisement.\n"
                f"Shot description: {shot['description_en']}\n"
                f"The ad's intended visual style: {style_desc}\n\n"
                f"Evaluate the cinematographic quality of this shot:\n"
                f"1. Does the lighting create an appropriate mood for the shot description?\n"
                f"2. Is the visual quality professional (not grainy, glitchy, or AI-artifact-heavy)?\n"
                f"3. Does the color palette fit the ad's intended style?\n\n"
                f"Score on a scale: 0.0 (poor quality), 0.5 (acceptable), 1.0 (excellent).\n"
                f"Answer in this format:\n"
                f"SCORE: X.X\n"
                f"EXPLANATION: (brief reasoning)"
            )
            response = self._query(prompt, frames)

            # Parse score
            s = 0.5
            for line in response.strip().split("\n"):
                if line.strip().upper().startswith("SCORE"):
                    try:
                        s = float(line.split(":")[-1].strip().split()[0])
                        s = max(0.0, min(1.0, s))
                    except (ValueError, IndexError):
                        s = 0.5
                    break

            total_score += s
            evaluated += 1
            results.append({
                "shot_id": shot["id"],
                "start_sec": round(scene[0], 3),
                "end_sec": round(scene[1], 3),
                "score": s,
                "response": response,
            })
            print(f"    Shot {shot['id']} atmosphere: {s:.2f}")

        score = total_score / evaluated if evaluated > 0 else 0
        return {"dimension": "Ads_Lens_Atmosphere", "score": score, "details": results}

    # ------------------------------------------------------------------
    # Dimension 9: Camera Motion (storyboard-based)
    # ------------------------------------------------------------------
    # Shot scale categories for VLM reference
    SCALE_GLOSSARY = {
        "大特": "extreme close-up — 极近景，强调眼睛、嘴唇、皮肤纹理或物体局部细节",
        "特写": "close-up — 面部或物体主要细节，占据画面大部分",
        "近景": "medium close-up / close shot — 头肩部，或胸部以上人物镜头",
        "中景": "medium shot — 半身镜头，通常腰部以上；有时也可更宽到膝部附近",
        "小全": "medium long shot / cowboy shot — 接近全身，人物基本完整，但环境不是主要重点",
        "全景": "full shot / wide shot — 人物全身或较完整场景，人物与环境关系清楚",
        "远景": "long shot / very wide shot — 远距离拍摄，人物较小，环境占比大",
    }

    def compute_ads_camera_motion(self) -> Dict:
        """Check shot scale (all shots) and camera movement (non-static shots).

        Two sub-checks per shot:
        1. Shot scale — is the framing distance correct? (all checkable shots)
        2. Camera movement — does the camera move as specified? (non-static only)

        Score = total passed sub-checks / total sub-checks.
        """
        present = self._get_present_shots()
        results = []
        total_checks = 0
        total_passed = 0

        for shot in present:
            scene = shot["aligned_scene"]
            # Only check shots explicitly marked for camera evaluation
            if not shot.get("check_camera", False):
                continue

            shot_scale = shot.get("shot_scale", "")
            camera = shot.get("camera", "static")
            cam_desc = shot.get("camera_description", "")

            check_scale = shot_scale not in ("", "footage", "vfx")
            check_movement = camera != "static"

            if not check_scale and not check_movement:
                continue

            frames = extract_frames(self.video_path, n_frames=6,
                                    start=scene[0], end=scene[1])
            if not frames:
                entry = {
                    "shot_id": shot["id"],
                    "start_sec": round(scene[0], 3),
                    "end_sec": round(scene[1], 3),
                }
                if check_scale:
                    entry["scale_expected"] = shot_scale
                    entry["scale_passed"] = False
                    entry["scale_reason"] = "No frames extracted"
                    total_checks += 1
                if check_movement:
                    entry["movement_expected"] = camera
                    entry["movement_passed"] = False
                    entry["movement_reason"] = "No frames extracted"
                    total_checks += 1
                results.append(entry)
                continue

            entry = {
                "shot_id": shot["id"],
                "start_sec": round(scene[0], 3),
                "end_sec": round(scene[1], 3),
            }

            # --- Sub-check 1: Shot scale ---
            if check_scale:
                scale_label = self.SCALE_GLOSSARY.get(shot_scale, shot_scale)
                glossary_lines = "\n".join(
                    f"- {k}: {v}" for k, v in self.SCALE_GLOSSARY.items()
                )
                scale_prompt = (
                    f"These {len(frames)} frames are from a single shot of an "
                    f"advertisement.\n\n"
                    f"Shot scale categories (from closest to farthest):\n"
                    f"{glossary_lines}\n\n"
                    f"The storyboard specifies this shot as: '{shot_scale}' ({scale_label})\n\n"
                    f"Based on the frames, does the framing distance match "
                    f"'{shot_scale}'? Focus only on how close or far the camera "
                    f"is from the main subject — ignore the content itself.\n"
                    f"Answer YES or NO first, then explain briefly."
                )
                is_yes, reason = self._query_yes_no(scale_prompt, frames)
                entry["scale_expected"] = shot_scale
                entry["scale_passed"] = is_yes
                entry["scale_reason"] = reason
                total_checks += 1
                if is_yes:
                    total_passed += 1
                print(f"    Shot {shot['id']} scale '{shot_scale}': "
                      f"{'PASS' if is_yes else 'FAIL'}")

            # --- Sub-check 2: Camera movement ---
            if check_movement:
                movement_prompt = (
                    f"These {len(frames)} frames are sequential frames from a "
                    f"single shot of an advertisement.\n\n"
                    f"The storyboard describes this shot's camera as:\n"
                    f"  \"{cam_desc}\"\n"
                    f"The expected camera movement type is: '{camera}'\n\n"
                    f"Camera movement reference:\n"
                    f"- tracking: camera follows a moving subject\n"
                    f"- tilt_down/tilt_up: camera rotates vertically\n"
                    f"- orbit: camera circles around the subject\n"
                    f"- pull_back/pull_out: camera moves away from subject\n"
                    f"- sweep: lateral sweeping movement\n"
                    f"- quick_cut/quick_cuts: rapid editing between angles\n"
                    f"- dynamic: energetic mixed movement\n"
                    f"- overhead: bird's eye / top-down view\n"
                    f"- pan_right: horizontal pan\n"
                    f"- through: camera moves through/into the scene\n\n"
                    f"Based on the frames, does the camera movement match "
                    f"'{camera}'? Focus only on how the viewpoint, perspective, "
                    f"and framing change between frames — not on the content.\n"
                    f"Answer YES or NO first, then explain briefly."
                )
                is_yes, reason = self._query_yes_no(movement_prompt, frames)
                entry["movement_expected"] = camera
                entry["movement_passed"] = is_yes
                entry["movement_reason"] = reason
                total_checks += 1
                if is_yes:
                    total_passed += 1
                print(f"    Shot {shot['id']} movement '{camera}': "
                      f"{'PASS' if is_yes else 'FAIL'}")

            results.append(entry)

        score = total_passed / total_checks if total_checks > 0 else 1.0
        return {"dimension": "Ads_Camera_Motion", "score": score,
                "passed": total_passed, "total_checks": total_checks,
                "details": results}

    # ------------------------------------------------------------------
    # Run all dimensions
    # ------------------------------------------------------------------
    def evaluate_all(self, dimensions: Optional[List[str]] = None) -> Dict:
        """Run all (or specified) evaluation dimensions.

        Args:
            dimensions: List of dimension names to run. None = all 8.

        Returns:
            Dict with scores per dimension and overall score.
        """
        all_dims = {
            "storyboard_adherence": self.compute_storyboard_adherence,
            "shot_specific_checks": self.compute_shot_specific_checks,
            "character_consistency": self.compute_character_consistency,
            "product_consistency": self.compute_product_consistency,
            "visual_style_unity": self.compute_visual_style_unity,
            "shot_continuity": self.compute_shot_continuity,
            "narrative_flow": self.compute_narrative_flow,
            "lens_atmosphere": self.compute_lens_atmosphere,
            "ads_camera_motion": self.compute_ads_camera_motion,
        }

        if dimensions:
            dims_to_run = {k: v for k, v in all_dims.items() if k in dimensions}
        else:
            dims_to_run = all_dims

        results = {}
        for name, fn in dims_to_run.items():
            print(f"\n{'='*60}")
            print(f"  Evaluating: {name}")
            print(f"{'='*60}")
            try:
                result = fn()
                results[name] = result
                print(f"  -> Score: {result['score']:.4f}")
            except Exception as e:
                print(f"  -> ERROR: {e}")
                results[name] = {"dimension": name, "score": 0, "error": str(e)}

        # Overall score = mean of all dimension scores
        scores = [r["score"] for r in results.values() if isinstance(r.get("score"), (int, float))]
        overall = sum(scores) / len(scores) if scores else 0

        return {
            "ad_id": self.config["ad_id"],
            "video_path": self.video_path,
            "overall_score": overall,
            "dimensions": results,
        }
