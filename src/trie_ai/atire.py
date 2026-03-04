from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any

from cosmos_reason_2_inference import CosmosReason2Inference
from trie_ai.reason2_runner import describe_frames_basic
from trie_ai.video_io import get_video_fps, get_video_fps, sample_frames_uniform

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

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
    
    try:
        return json.loads(m.group(0))
    except Exception:
        pass

    return {"error": "Model output was not valid JSON, and JSON extraction failed.", "raw_output": text}

@dataclass
class AtireConfig:
    """
    Backends:
      - stub: no model call, creates conservative placeholder
      - vllm_inprocess: uses CosmosReason2Inference (vLLM) directly in-process
    """
    mode: str = "stub"  # "stub" | "vllm_inprocess"
    model_path: str = "nvidia/Cosmos-Reason2-2B"
    workdir: str = "results/tmp_frames"

    # Generation params
    max_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    seed: int = 0
    timeout_s: int = 120

    # Local vLLM (in-process) knobs
    nframes: int = 8
    max_model_len: int = 8192
    dtype: str = "half"
    enforce_eager: bool = True
    gpu_memory_utilization: float = 0.90


class Atire:
    def __init__(self, config: AtireConfig | None = None, *, auto_load: bool = False) -> None:
        self.config = config or AtireConfig()
        if self.config.mode not in ("stub", "vllm_inprocess"):
            raise ValueError(f"Unsupported mode: {self.config.mode}")
        self._inference = None
        if self.config.mode == "vllm_inprocess":
            self._inference = CosmosReason2Inference(
                model_path=self.config.model_path,
                nframes=self.config.nframes,
                max_tokens=self.config.max_tokens,
                max_model_len=self.config.max_model_len,
                dtype=self.config.dtype,
                enforce_eager=self.config.enforce_eager,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                seed=self.config.seed,
            )
        self._loaded = False

        if auto_load:
            self._load_model()

    def _load_model(self) -> None:
        ok = self._inference.load_model()
        if not ok:
            raise RuntimeError("Failed to load Cosmos Reason 2 model.")
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _metadata(self, model, video_id, frame_desc, fps) -> dict[str, Any]:
        return {
            "metadata": {
                "model": model,
                "video_id": video_id,
                "frame_sampling": {"strategy": "uniform", "count": len(frame_desc), "fps": fps},
                "generated_at": _now_iso(),
            }
        }
    def _run_report_stub(self, video_id: str, fps: float, frame_desc: list[dict[str, Any]]) -> dict:
        evidence = [f"{f['frame_index']})" for f in frame_desc[:3]]
        stub_rep = {
            "summary": "Potential traffic interaction detected; report generated from sampled frames (stub mode).",
            "actors_list": [{"id": "veh_1", "type": "car"}],
            "events_timeline": [{
                "t_start": 0.0,
                "t_end": max([f["timestamp_s"] for f in frame_desc] or [0.0]),
                "event": "Scene observed; insufficient detail for precise event segmentation (stub).",
                "evidence": evidence
            }],
            "uncertainties": ["Running in stub mode."],
            "risk_assessment": {"level": "medium", "why": "Stub mode cannot fully assess speed/distance; flagged for review."},
        }
        stub_rep["metadata"] = self._metadata(model="stub", video_id=video_id, frame_desc=frame_desc, fps=fps)["metadata"]
        return stub_rep

    def _run_report_vllm_inprocess(
        self,
        *,
        video_id: str,
        fps: float,
        frame_desc: list[dict[str, Any]],
        system_prompt: str,
        user_prompt: str,
        video_path: str | Path | None = None,
    ) -> dict:
        if video_path is None:
            raise ValueError("vllm_inprocess mode requires video_path.")

        # Provide frame indices/timestamps context to keep outputs grounded.
        frame_index_block = "\n".join([f"- idx={f['frame_index']} t={f['timestamp_s']}s" for f in frame_desc])
        full_user_text = (
            f"{user_prompt}\n\n"
            f"Context:\nvideo_id={video_id}\nvideo_fps={fps}\nSampled frames:\n{frame_index_block}\n\n"
            "Return ONLY valid JSON."
        )
        text = self.query(
            video_path=str(video_path) if video_path else "",
            question=full_user_text,
            system_prompt=system_prompt,
        )
        if text is None:
            raise RuntimeError("Inference returned no response.")
        # Parse out JSON from the model's text response, being defensive about formatting issues.
        rep = _extract_json(text)
        rep["metadata"] = self._metadata(model=self.config.model_path or self.config.model_size, video_id=video_id, frame_desc=frame_desc, fps=fps)["metadata"]
        return rep

    def generate_report(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        video_path: str | Path | None = None,
    ) -> dict:
        video_path = Path(video_path) if video_path else None
        if video_path is None:
            raise ValueError("vllm_inprocess mode requires video_path.")
        # Get video description from frames and metadata, then generate report using the configured mode
        video_id = video_path.stem if video_path else "unknown_video"
        fps = get_video_fps(str(video_path))
        samples = sample_frames_uniform(str(video_path), self.config.workdir, num_frames=self.config.nframes)
        frame_tuples = [(s.index, s.timestamp_s, s.path) for s in samples]
        frame_desc = describe_frames_basic(frame_tuples)

        # Generate report using the configured mode
        if self.config.mode == "vllm_inprocess":
            return self._run_report_vllm_inprocess(
                video_id=video_id,
                fps=fps,
                frame_desc=frame_desc,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                video_path=video_path,
            )
        return self._run_report_stub(video_id=video_id, fps=fps, frame_desc=frame_desc)

    def query(
        self,
        *,
        video_path: str | Path,
        question: str,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        if not self._loaded:
            self._load_model()

        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")

        response = self._inference.query(
            video_path=str(path),
            question=question,
            system_prompt=system_prompt,
        )
        if response is None:
            raise RuntimeError("Inference returned no response.")
        return response

    def close(self) -> None:
        if self._inference:
            self._inference.close()
        self._loaded = False

    def __enter__(self) -> "Atire":
        # if not self._loaded:
        #     self._load_model()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

