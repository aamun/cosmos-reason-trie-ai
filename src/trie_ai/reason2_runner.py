from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import base64
import json
import os
import re
from typing import Any, Iterable

import requests


@dataclass
class Reason2Config:
    """
    Backends:
      - stub: no model call, creates conservative placeholder
      - vllm_openai: calls a vLLM OpenAI-compatible endpoint: POST http://HOST:PORT/v1/chat/completions
      - vllm_inprocess: uses CosmosReason2Inference (vLLM) directly in-process
    """
    mode: str = "stub"  # "stub" | "vllm_openai" | "vllm_inprocess"

    # Model selection
    model_size: str = "2b"  # "2b" | "8b"
    model_id: str | None = None  # override full HF id e.g. "nvidia/Cosmos-Reason2-2B"

    # Endpoint auth (Nebius or local vLLM)
    endpoint: str | None = None  # "PUBLICIP:8000" or "localhost:8000"
    api_key: str | None = None

    # Generation params
    max_tokens: int = 768
    temperature: float = 0.2
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    seed: int = 0
    timeout_s: int = 120

    # Local vLLM (in-process) knobs
    nframes: int = 8
    max_model_len: int = 2048
    dtype: str = "half"
    enforce_eager: bool = True
    gpu_memory_utilization: float = 0.90


def _default_model_id(model_size: str) -> str:
    # Matches Nebius guide config and model names. :contentReference[oaicite:3]{index=3}
    size = model_size.lower()
    if size in ("2b", "2"):
        return "nvidia/Cosmos-Reason2-2B"
    if size in ("8b", "8"):
        return "nvidia/Cosmos-Reason2-8B"
    raise ValueError(f"Unknown model_size: {model_size} (use '2b' or '8b')")


def describe_frames_basic(frames: list[tuple[int, float, Path]]) -> list[dict[str, Any]]:
    # Minimal descriptors; we will also send the actual images to the model.
    return [
        {"frame_index": idx, "timestamp_s": round(ts, 2), "path": str(path)}
        for idx, ts, path in frames
    ]


def _img_to_data_url_jpg(path: Path) -> str:
    b = path.read_bytes()
    enc = base64.b64encode(b).decode("utf-8")
    return f"data:image/jpeg;base64,{enc}"


def _extract_json(text: str) -> dict[str, Any]:
    """
    Cosmos outputs should be JSON (we request it), but be defensive:
    - try direct json.loads
    - else attempt to grab the first {...} block
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Model output was not JSON and no JSON object could be extracted.")
    return json.loads(m.group(0))


def run_reason2_report_stub(video_id: str, fps: float, frame_desc: list[dict[str, Any]]) -> dict:
    evidence = [f"frame@{f['timestamp_s']}s(idx={f['frame_index']})" for f in frame_desc[:3]]
    return {
        "video_id": video_id,
        "summary": "Potential traffic interaction detected; report generated from sampled frames (stub mode).",
        "actors": [{"id": "veh_1", "type": "car", "notes": "uncertain"}],
        "timeline": [{
            "t_start": 0.0,
            "t_end": max([f["timestamp_s"] for f in frame_desc] or [0.0]),
            "event": "Scene observed; insufficient detail for precise event segmentation (stub).",
            "evidence": evidence
        }],
        "risk_assessment": {"level": "medium", "why": "Stub mode cannot fully assess speed/distance; flagged for review."},
        "causal_chain": [{"cause": "Limited visual context in stub mode", "effect": "Conservative risk estimate"}],
        "metadata": {
            "model": "stub",
            "frame_sampling": {"strategy": "uniform", "count": len(frame_desc), "fps": fps},
            "generated_at": ""
        }
    }


def run_reason2_report_vllm_openai(
    *,
    config: Reason2Config,
    video_id: str,
    fps: float,
    frame_desc: list[dict[str, Any]],
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    """
    Sends sampled frames as 'image_url' items BEFORE text ("media-first ordering"). :contentReference[oaicite:4]{index=4}
    Compatible with:
      - Nebius vLLM endpoint
      - local vLLM server
    """
    endpoint = config.endpoint or os.getenv("VLLM_ENDPOINT")
    api_key = config.api_key or os.getenv("VLLM_API_KEY")
    if not endpoint:
        raise ValueError("Missing vLLM endpoint. Set config.endpoint or env VLLM_ENDPOINT (e.g., localhost:8000).")
    if not api_key:
        raise ValueError("Missing vLLM API key. Set config.api_key or env VLLM_API_KEY.")

    model_id = config.model_id or _default_model_id(config.model_size)

    # Build content: images first, then text (recommended prompting pattern). :contentReference[oaicite:5]{index=5}
    content: list[dict[str, Any]] = []
    for f in frame_desc:
        p = Path(f["path"])
        content.append({
            "type": "image_url",
            "image_url": {"url": _img_to_data_url_jpg(p)}
        })

    # Provide frame indices/timestamps context inside text too, to help grounding.
    frame_index_block = "\n".join([f"- idx={f['frame_index']} t={f['timestamp_s']}s" for f in frame_desc])
    full_user_text = (
        f"{user_prompt}\n\n"
        f"Context:\nvideo_id={video_id}\nvideo_fps={fps}\nSampled frames:\n{frame_index_block}\n\n"
        "Return ONLY valid JSON."
    )

    content.append({"type": "text", "text": full_user_text})

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }

    url = f"http://{endpoint}/v1/chat/completions"
    r = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json=payload,
        timeout=config.timeout_s,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"vLLM error {r.status_code}: {r.text[:500]}")

    data = r.json()
    text = data["choices"][0]["message"]["content"]
    rep = _extract_json(text)

    # Attach metadata (don’t overwrite if already present)
    rep.setdefault("metadata", {})
    rep["metadata"].setdefault("model", model_id)
    rep["metadata"].setdefault("frame_sampling", {"strategy": "uniform", "count": len(frame_desc), "fps": fps})
    rep["metadata"].setdefault("generated_at", "")
    return rep


def run_reason2_report_vllm_inprocess(
    *,
    config: Reason2Config,
    video_id: str,
    fps: float,
    frame_desc: list[dict[str, Any]],
    system_prompt: str,
    user_prompt: str,
    video_path: str | Path | None,
) -> dict[str, Any]:
    """
    Run Cosmos Reason 2 locally using vLLM in-process.
    """
    if video_path is None:
        raise ValueError("vllm_inprocess mode requires video_path.")

    model_id = config.model_id or _default_model_id(config.model_size)

    # Import lazily to avoid heavy deps when not used.
    from cosmos_reason_2_inference import CosmosReason2Inference

    inference = CosmosReason2Inference(
        model_path=model_id,
        nframes=config.nframes,
        max_tokens=config.max_tokens,
        max_model_len=config.max_model_len,
        enforce_eager=config.enforce_eager,
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        dtype=config.dtype,
        gpu_memory_utilization=config.gpu_memory_utilization,
        seed=config.seed,
    )

    # Provide frame indices/timestamps context to keep outputs grounded.
    frame_index_block = "\n".join([f"- idx={f['frame_index']} t={f['timestamp_s']}s" for f in frame_desc])
    full_user_text = (
        f"{user_prompt}\n\n"
        f"Context:\nvideo_id={video_id}\nvideo_fps={fps}\nSampled frames:\n{frame_index_block}\n\n"
        "Return ONLY valid JSON."
    )

    try:
        ok = inference.load_model()
        if not ok:
            raise RuntimeError("Failed to load Cosmos Reason 2 model.")

        text = inference.query(
            video_path=str(video_path),
            question=full_user_text,
            system_prompt=system_prompt,
        )
        print(f"Raw model output:\n{text}\n{'-'*40}")
        if text is None:
            raise RuntimeError("Inference returned no text response.")
        rep = _extract_json(text)
    finally:
        # Free GPU/CPU resources ASAP
        try:
            inference.close()
        except Exception:
            pass

    rep.setdefault("metadata", {})
    rep["metadata"].setdefault("model", model_id)
    rep["metadata"].setdefault("frame_sampling", {"strategy": "uniform", "count": len(frame_desc), "fps": fps})
    rep["metadata"].setdefault("generated_at", "")
    return rep


def generate_report(
    config: Reason2Config,
    video_id: str,
    fps: float,
    frame_desc: list[dict[str, Any]],
    *,
    system_prompt: str,
    user_prompt: str,
    video_path: str | Path | None = None,
) -> dict:
    if config.mode == "vllm_openai":
        return run_reason2_report_vllm_openai(
            config=config,
            video_id=video_id,
            fps=fps,
            frame_desc=frame_desc,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    if config.mode == "vllm_inprocess":
        return run_reason2_report_vllm_inprocess(
            config=config,
            video_id=video_id,
            fps=fps,
            frame_desc=frame_desc,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            video_path=video_path,
        )
    return run_reason2_report_stub(video_id=video_id, fps=fps, frame_desc=frame_desc)
