"""
OpenRouter VLM client for ads evaluation.

Supports sending images (base64-encoded frames) + text prompts to
vision-language models via the OpenRouter API.

Default model: anthropic/claude-sonnet-4-6 (vision-capable, high quality).
"""

import os
import base64
import json
import time
import requests
from typing import List, Optional


DEFAULT_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"
API_URL = "https://openrouter.ai/api/v1/chat/completions"


def _encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _frame_to_base64(frame, max_dim=768) -> str:
    """Convert a numpy array (H, W, 3) or PIL Image to base64 JPEG.

    Resizes to max_dim on the longest side to keep payload under API limits.
    Uses JPEG instead of PNG for ~5-10x smaller payloads.
    """
    from PIL import Image
    import io

    if not isinstance(frame, Image.Image):
        frame = Image.fromarray(frame)
    # Resize if too large
    w, h = frame.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        frame = frame.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    frame.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def query_vlm(
    prompt: str,
    frames: Optional[List] = None,
    image_paths: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0,
    max_retries: int = 5,
) -> str:
    """Send a vision+text query to a VLM via OpenRouter.

    Args:
        prompt: Text prompt/question.
        frames: List of numpy arrays (H,W,3) or PIL Images to include as images.
        image_paths: List of image file paths to include.
        system_prompt: Optional system prompt.
        api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
        model: OpenRouter model identifier.
        max_tokens: Max response tokens.
        temperature: Sampling temperature.
        max_retries: Number of retries on transient failures.

    Returns:
        Model response text.
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("No OpenRouter API key provided. Set OPENROUTER_API_KEY env var.")

    # Build content array with images + text
    content = []

    # Add frames as base64 images
    if frames:
        for frame in frames:
            b64 = _frame_to_base64(frame)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

    # Add image files
    if image_paths:
        for path in image_paths:
            b64 = _encode_image_to_base64(path)
            ext = path.rsplit(".", 1)[-1].lower()
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"}
            })

    # Add text prompt
    content.append({"type": "text", "text": prompt})

    # Build messages — use plain string if no images (some providers reject array for text-only)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    has_images = bool(frames) or bool(image_paths)
    if has_images:
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=180)
            # Log response body on non-2xx for debugging
            if response.status_code >= 400:
                body = response.text[:500]
                print(f"  [OpenRouter] HTTP {response.status_code} response body: {body}")
                # Log request metadata (not content) for debugging
                n_images = sum(1 for m in messages for c in (m.get("content", []) if isinstance(m.get("content"), list) else []) if isinstance(c, dict) and c.get("type") == "image_url")
                print(f"  [OpenRouter] Request: model={model}, n_images={n_images}, max_tokens={max_tokens}")
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except (requests.exceptions.RequestException, KeyError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)  # 2, 4, 8, 16...
                print(f"  [OpenRouter] Retry {attempt + 1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise RuntimeError(f"OpenRouter API failed after {max_retries} attempts: {e}") from e


def query_vlm_yes_no(
    prompt: str,
    frames: Optional[List] = None,
    image_paths: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    **kwargs,
) -> tuple:
    """Query VLM with a yes/no question. Returns (bool, str reason)."""
    response = query_vlm(
        prompt=prompt + "\n\nAnswer YES or NO first, then give a brief reason.",
        frames=frames,
        image_paths=image_paths,
        system_prompt=system_prompt,
        **kwargs,
    )
    # Strip markdown formatting (**YES**, *yes*, etc.) and leading punctuation
    cleaned = response.strip().lower()
    cleaned = cleaned.lstrip("*_#> ")
    is_yes = cleaned.startswith("yes")
    return is_yes, response
