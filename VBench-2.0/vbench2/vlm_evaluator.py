"""
VLM-based VBench 2.0 dimension evaluator using Claude Sonnet 4.6 via OpenRouter.

Replaces the LLaVA-Video-7B + Qwen2.5 judge pipeline with a single VLM API call.
No GPU needed — all evaluation goes through OpenRouter API.

Supports all 13 VLM-based dimensions:
  Composition, Dynamic_Spatial_Relationship, Dynamic_Attribute,
  Motion_Order_Understanding, Human_Interaction, Complex_Landscape,
  Complex_Plot, Camera_Motion, Motion_Rationality, Mechanics,
  Thermotics, Material, Human_Clothes
"""

import json
import os
import re
import numpy as np
from PIL import Image
import cv2


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _extract_frames(video_path, max_frames=16):
    """Extract uniformly sampled frames from a video as PIL Images."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int).tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def _extract_first_half_frames(video_path, max_frames=8):
    """Extract frames from the first half of the video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    half = total // 2
    indices = np.linspace(0, half - 1, min(max_frames, half), dtype=int).tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def _extract_last_frame(video_path):
    """Extract the last frame of the video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None


def _query_vlm_yesno(prompt, frames, api_key, model="qwen/qwen3-vl-235b-a22b-instruct"):
    """Send frames + prompt to VLM, return (is_yes: bool, response: str)."""
    from vbench2.openrouter_vlm import query_vlm
    response = query_vlm(
        prompt=prompt + "\n\nAnswer YES or NO first, then give a brief reason.",
        frames=frames,
        api_key=api_key,
        model=model,
        max_tokens=256,
        temperature=0,
    )
    if response is None:
        return False, "No response from VLM"
    cleaned = response.strip().lower().lstrip("*_#> ")
    is_yes = cleaned.startswith("yes")
    return is_yes, response


def _query_vlm_text(prompt, frames, api_key, model="qwen/qwen3-vl-235b-a22b-instruct", max_tokens=1024):
    """Send frames + prompt to VLM, return response text."""
    from vbench2.openrouter_vlm import query_vlm
    return query_vlm(
        prompt=prompt,
        frames=frames,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=0,
    )


def load_dimension_info(json_dir, dimension, lang='en'):
    """Load VBench dimension info — same interface as vbench2.utils.load_dimension_info."""
    video_list = []
    prompt_dict_ls = []
    full_prompt_list = _load_json(json_dir)
    for prompt_dict in full_prompt_list:
        if dimension in prompt_dict['dimension'][0].lower() and 'video_list' in prompt_dict:
            prompt = prompt_dict[f'prompt_{lang}']
            cur_video_list = prompt_dict['video_list'] if isinstance(prompt_dict['video_list'], list) else [prompt_dict['video_list']]
            video_list += cur_video_list
            if 'auxiliary_info' in prompt_dict:
                prompt_dict_ls.append({'prompt': prompt, 'video_list': cur_video_list, 'auxiliary_info': prompt_dict['auxiliary_info']})
            else:
                prompt_dict_ls.append({'prompt': prompt, 'video_list': cur_video_list})
    return video_list, prompt_dict_ls


# ── Dimension implementations ──────────────────────────────────────────────

def compute_composition(json_dir, device, submodules_dict, **kwargs):
    """Composition: checks if video contains described elements."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='composition', lang='en')

    video_results = []
    for prompt_dict in prompt_dict_ls:
        questions = prompt_dict['auxiliary_info']['question']
        num_judge0 = prompt_dict['auxiliary_info']['judge'][0]  # check single creature first
        num_judge1 = prompt_dict['auxiliary_info']['judge'][1]  # strict: all must match
        question_num = len(questions)

        for video_path in prompt_dict['video_list']:
            frames = _extract_frames(video_path, max_frames=16)
            score = 0
            valid = True

            if num_judge0:
                # First check: only one creature?
                is_yes, _ = _query_vlm_yesno(
                    "Does this video show only one creature/entity?", frames, api_key)
                if not is_yes:
                    valid = False
                else:
                    for q in questions:
                        is_yes, _ = _query_vlm_yesno(
                            f"Does this video contain: {q}?", frames, api_key)
                        if is_yes:
                            score += 1
            else:
                for q in questions:
                    is_yes, _ = _query_vlm_yesno(
                        f"Does this video contain: {q}?", frames, api_key)
                    if is_yes:
                        score += 1

            if valid:
                if num_judge1:
                    sco = 1 if score == question_num else 0
                else:
                    sco = score / question_num
            else:
                sco = 1 / question_num

            video_results.append({'video_path': video_path, 'video_results': sco})

    all_results = sum(d['video_results'] for d in video_results) / len(video_results)
    return all_results, video_results


def compute_dynamic_attribute(json_dir, device, submodules_dict, **kwargs):
    """Dynamic_Attribute: checks temporal attribute changes via yes/no questions."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='dynamic_attribute', lang='en')

    video_results = []
    for prompt_dict in prompt_dict_ls:
        questions = prompt_dict['auxiliary_info']
        for video_path in prompt_dict['video_list']:
            frames = _extract_frames(video_path, max_frames=16)
            flag = True
            for q in questions:
                is_yes, _ = _query_vlm_yesno(q, frames, api_key)
                if not is_yes:
                    flag = False
            sco = 1 if flag else 0
            video_results.append({'video_path': video_path, 'video_results': sco})

    all_results = sum(d['video_results'] for d in video_results) / len(video_results)
    return all_results, video_results


def compute_dynamic_spatial_relationship(json_dir, device, submodules_dict, **kwargs):
    """Dynamic_Spatial_Relationship: checks spatial relations at different time points."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='dynamic_spatial_relationship', lang='en')

    video_results = []
    for prompt_dict in prompt_dict_ls:
        questions = prompt_dict['auxiliary_info']
        for video_path in prompt_dict['video_list']:
            flag = True
            for i, q in enumerate(questions):
                if i == 0:
                    # First question: check first half of video
                    frames = _extract_first_half_frames(video_path, max_frames=8)
                else:
                    # Second question: check last frame
                    last_frame = _extract_last_frame(video_path)
                    frames = [last_frame] if last_frame else []
                if frames:
                    is_yes, _ = _query_vlm_yesno(q, frames, api_key)
                    if not is_yes:
                        flag = False
            sco = 1 if flag else 0
            video_results.append({'video_path': video_path, 'video_results': sco})

    all_results = sum(d['video_results'] for d in video_results) / len(video_results)
    return all_results, video_results


def compute_motion_rationality(json_dir, device, submodules_dict, **kwargs):
    """Motion_Rationality: checks physics/motion correctness via yes/no questions."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='motion_rationality', lang='en')

    video_results = []
    for prompt_dict in prompt_dict_ls:
        questions = prompt_dict['auxiliary_info']
        for video_path in prompt_dict['video_list']:
            frames = _extract_frames(video_path, max_frames=16)
            score = 0
            for q in questions:
                is_yes, _ = _query_vlm_yesno(q, frames, api_key)
                if is_yes:
                    score += 1
            sco = 1 if score == len(questions) else 0
            video_results.append({'video_path': video_path, 'video_results': sco})

    all_results = sum(d['video_results'] for d in video_results) / len(video_results)
    return all_results, video_results


def compute_mechanics(json_dir, device, submodules_dict, **kwargs):
    """Mechanics: checks physics phenomena with validity gate."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='mechanics', lang='en')

    video_results = []
    for prompt_dict in prompt_dict_ls:
        questions = prompt_dict['auxiliary_info']
        for video_path in prompt_dict['video_list']:
            frames = _extract_frames(video_path, max_frames=16)
            valid = True
            flag = True

            if len(questions) == 1:
                is_yes, _ = _query_vlm_yesno(questions[0], frames, api_key)
                if not is_yes:
                    flag = False
            else:
                for i, q in enumerate(questions):
                    is_yes, _ = _query_vlm_yesno(q, frames, api_key)
                    if i == 0 and not is_yes:
                        valid = False
                    elif i == 1 and not is_yes:
                        flag = False

            if not valid:
                video_results.append({'video_path': video_path, 'video_results': -1})
            elif flag:
                video_results.append({'video_path': video_path, 'video_results': 1})
            else:
                video_results.append({'video_path': video_path, 'video_results': 0})

    score = sum(d['video_results'] for d in video_results if d['video_results'] != -1)
    num = sum(1 for d in video_results if d['video_results'] != -1)
    all_results = score / num if num > 0 else 0
    return all_results, video_results


def compute_thermotics(json_dir, device, submodules_dict, **kwargs):
    """Thermotics: same pattern as mechanics (validity gate + check)."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='thermotics', lang='en')

    video_results = []
    for prompt_dict in prompt_dict_ls:
        questions = prompt_dict['auxiliary_info']
        for video_path in prompt_dict['video_list']:
            frames = _extract_frames(video_path, max_frames=16)
            valid = True
            flag = True

            if len(questions) == 1:
                is_yes, _ = _query_vlm_yesno(questions[0], frames, api_key)
                if not is_yes:
                    flag = False
            else:
                for i, q in enumerate(questions):
                    is_yes, _ = _query_vlm_yesno(q, frames, api_key)
                    if i == 0 and not is_yes:
                        valid = False
                    elif i == 1 and not is_yes:
                        flag = False

            if not valid:
                video_results.append({'video_path': video_path, 'video_results': -1})
            elif flag:
                video_results.append({'video_path': video_path, 'video_results': 1})
            else:
                video_results.append({'video_path': video_path, 'video_results': 0})

    score = sum(d['video_results'] for d in video_results if d['video_results'] != -1)
    num = sum(1 for d in video_results if d['video_results'] != -1)
    all_results = score / num if num > 0 else 0
    return all_results, video_results


def compute_material(json_dir, device, submodules_dict, **kwargs):
    """Material: same pattern as mechanics/thermotics."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='material', lang='en')

    video_results = []
    for prompt_dict in prompt_dict_ls:
        questions = prompt_dict['auxiliary_info']
        for video_path in prompt_dict['video_list']:
            frames = _extract_frames(video_path, max_frames=16)
            valid = True
            flag = True

            if len(questions) == 1:
                is_yes, _ = _query_vlm_yesno(questions[0], frames, api_key)
                if not is_yes:
                    flag = False
            else:
                for i, q in enumerate(questions):
                    is_yes, _ = _query_vlm_yesno(q, frames, api_key)
                    if i == 0 and not is_yes:
                        valid = False
                    elif i == 1 and not is_yes:
                        flag = False

            if not valid:
                video_results.append({'video_path': video_path, 'video_results': -1})
            elif flag:
                video_results.append({'video_path': video_path, 'video_results': 1})
            else:
                video_results.append({'video_path': video_path, 'video_results': 0})

    score = sum(d['video_results'] for d in video_results if d['video_results'] != -1)
    num = sum(1 for d in video_results if d['video_results'] != -1)
    all_results = score / num if num > 0 else 0
    return all_results, video_results


def compute_human_clothes(json_dir, device, submodules_dict, **kwargs):
    """Human_Clothes: checks person consistency and clothing consistency."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='human_clothes', lang='en')

    questions = [
        "Is there only one person in the video throughout?",
        "Is the person in the video the same throughout?",
        "Does the clothes of the person in the video (color, texture) remain consistent throughout?",
    ]

    video_results = []
    for prompt_dict in prompt_dict_ls:
        for video_path in prompt_dict['video_list']:
            frames = _extract_frames(video_path, max_frames=16)
            valid = True
            flag = True

            for i, q in enumerate(questions):
                is_yes, _ = _query_vlm_yesno(q, frames, api_key)
                if i == 0 and not is_yes:
                    valid = False
                    break
                elif i != 0 and not is_yes:
                    flag = False

            if not valid:
                video_results.append({'video_path': video_path, 'video_results': -1})
            elif flag:
                video_results.append({'video_path': video_path, 'video_results': 1})
            else:
                video_results.append({'video_path': video_path, 'video_results': 0})

    score = sum(d['video_results'] for d in video_results if d['video_results'] != -1)
    num = sum(1 for d in video_results if d['video_results'] != -1)
    all_results = score / num if num > 0 else 0
    return all_results, video_results


def compute_camera_motion(json_dir, device, submodules_dict, **kwargs):
    """Camera_Motion: detect camera motion type from video frames."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='camera_motion', lang='en')

    video_results = []
    for prompt_dict in prompt_dict_ls:
        label = prompt_dict['auxiliary_info']  # e.g. "zoom_in", "pan_left"
        for video_path in prompt_dict['video_list']:
            frames = _extract_frames(video_path, max_frames=16)
            prompt = (
                f"Analyze the camera motion in this video. "
                f"Possible camera motions: static, pan_left, pan_right, tilt_up, tilt_down, "
                f"zoom_in, zoom_out, orbits, oblique. "
                f"Which camera motion(s) are present? List them separated by commas. "
                f"Only list the motion types, nothing else."
            )
            response = _query_vlm_text(prompt, frames, api_key, max_tokens=128)
            response_lower = response.lower().replace("-", "_").replace(" ", "_")
            # Check if expected label is in the response
            label_clean = label.lower().replace("-", "_")
            video_score = 1.0 if label_clean in response_lower else 0.0
            video_results.append({'video_path': video_path, 'video_results': video_score})

    all_results = sum(d['video_results'] for d in video_results) / len(video_results)
    return all_results, video_results


def _split_numbered_list(text):
    """Split text like '1. xxx 2. yyy' into list of items."""
    parts = re.split(r'\d+\.\s*', text)
    return [p.strip() for p in parts if p.strip()]


def compute_complex_plot(json_dir, device, submodules_dict, **kwargs):
    """Complex_Plot: check if video matches ordered plot points."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='complex_plot', lang='en')

    video_results = []
    for prompt_dict in prompt_dict_ls:
        ground_truth = prompt_dict['auxiliary_info']  # list of plot points
        length = len(ground_truth)

        for video_path in prompt_dict['video_list']:
            frames = _extract_frames(video_path, max_frames=16)

            # Ask VLM to describe the plot
            template = "; ".join([f"{i+1}. " for i in range(length)])
            prompt = (
                f"Describe the plot/story of this video step by step. "
                f"Break it into exactly {length} key plot points. "
                f"Use this format: {template}"
            )
            response = _query_vlm_text(prompt, frames, api_key, max_tokens=1024)

            # Parse response into plot points
            plot_points = _split_numbered_list(response)

            # Compare each plot point with ground truth
            score = 0
            for q, gt_item in enumerate(ground_truth):
                if q < len(plot_points):
                    desc = plot_points[q]
                else:
                    desc = response  # fallback: compare against full response

                # Use VLM as judge
                judge_prompt = (
                    f"Does the following description contain the key elements of the reference?\n"
                    f"Reference: {gt_item}\n"
                    f"Description: {desc}\n"
                    f"Similar semantics should be accepted. Focus on the core action/event, "
                    f"not exact wording or minor details."
                )
                is_yes, _ = _query_vlm_yesno(judge_prompt, [], api_key)
                if is_yes:
                    score += 1
                else:
                    break  # ordered: stop at first mismatch

            video_results.append({
                'video_path': video_path,
                'video_results': score / length,
            })

    all_results = sum(d['video_results'] for d in video_results) / len(video_results)
    return all_results, video_results


def compute_complex_landscape(json_dir, device, submodules_dict, **kwargs):
    """Complex_Landscape: check if video matches ordered landscape descriptions."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='complex_landscape', lang='en')

    video_results = []
    for prompt_dict in prompt_dict_ls:
        ground_truth = prompt_dict['auxiliary_info']
        length = len(ground_truth)

        for video_path in prompt_dict['video_list']:
            frames = _extract_frames(video_path, max_frames=16)

            # Describe landscape
            prompt = (
                f"Describe the landscape and scenery shown in this video in detail. "
                f"Break it into exactly 5 key landscape descriptions. "
                f"Use numbered format: 1. ; 2. ; 3. ; 4. ; 5. "
            )
            response = _query_vlm_text(prompt, frames, api_key, max_tokens=1024)
            plot_points = _split_numbered_list(response)

            score = 0
            for q, gt_item in enumerate(ground_truth):
                if q < len(plot_points):
                    desc = plot_points[q]
                else:
                    desc = response

                judge_prompt = (
                    f"Does the following description contain the landscape element described in the reference?\n"
                    f"Reference: {gt_item}\n"
                    f"Description: {desc}\n"
                    f"Focus only on the landscape elements mentioned in the reference."
                )
                is_yes, _ = _query_vlm_yesno(judge_prompt, [], api_key)
                if is_yes:
                    score += 1
                else:
                    break

            video_results.append({
                'video_path': video_path,
                'video_results': score / length,
            })

    all_results = sum(d['video_results'] for d in video_results) / len(video_results)
    return all_results, video_results


def compute_human_interaction(json_dir, device, submodules_dict, **kwargs):
    """Human_Interaction: check if video shows described human interaction."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='human_interaction', lang='en')

    video_results = []
    for prompt_dict in prompt_dict_ls:
        ground_truth = prompt_dict['prompt'].strip()

        for video_path in prompt_dict['video_list']:
            frames = _extract_frames(video_path, max_frames=16)
            score = 0

            # Check 1: Does the interaction match?
            judge_prompt = (
                f"Does this video show the following human interaction: \"{ground_truth}\"?\n"
                f"Focus on the specific interaction described, not just the presence of people. "
                f"Be strict: 'holding a tea' is NOT the same as 'drinking tea'. "
                f"'handing a glass' is NOT the same as 'clinking glasses'."
            )
            is_yes, _ = _query_vlm_yesno(judge_prompt, frames, api_key)
            if is_yes:
                score += 1

            # Check 2: Are there multiple people?
            is_yes, _ = _query_vlm_yesno(
                "Does this video show more than one person?", frames, api_key)
            if is_yes:
                score += 1

            video_results.append({
                'video_path': video_path,
                'video_results': 1 if score == 2 else 0,
            })

    all_results = sum(d['video_results'] for d in video_results) / len(video_results)
    return all_results, video_results


def compute_motion_order_understanding(json_dir, device, submodules_dict, **kwargs):
    """Motion_Order_Understanding: check if actions happen in correct order."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='motion_order_understanding', lang='en')

    video_results = []
    for prompt_dict in prompt_dict_ls:
        ground_truth = prompt_dict['auxiliary_info']  # [action1, action2]

        for video_path in prompt_dict['video_list']:
            frames = _extract_frames(video_path, max_frames=16)

            # Ask VLM to describe the action order
            prompt = (
                "Describe the sequence of actions in this video in order. "
                "Break it into exactly 2 steps. "
                "Format: 1. [first action]; 2. [second action]"
            )
            response = _query_vlm_text(prompt, frames, api_key, max_tokens=512)

            parts = response.split('2.')
            if len(parts) < 2:
                video_results.append({'video_path': video_path, 'video_results': -1})
                continue

            score = 0
            for q in range(2):
                judge_prompt = (
                    f"Is the following description consistent with the action \"{ground_truth[q]}\"?\n"
                    f"Description: {parts[q]}\n"
                    f"Similar semantics count as consistent. "
                    f"Active and passive expressions can be considered consistent. "
                    f"But 'holding a glass of water' is NOT consistent with 'drinking water'."
                )
                is_yes, _ = _query_vlm_yesno(judge_prompt, [], api_key)
                if is_yes:
                    score += 1

            video_results.append({
                'video_path': video_path,
                'video_results': 1 if score == 2 else 0,
            })

    score = sum(d['video_results'] for d in video_results if d['video_results'] != -1)
    num = sum(1 for d in video_results if d['video_results'] != -1)
    count = sum(1 for d in video_results if d['video_results'] == -1)
    if count > 0.9 * len(video_results):
        return 0, video_results
    all_results = score / num if num > 0 else 0
    return all_results, video_results
