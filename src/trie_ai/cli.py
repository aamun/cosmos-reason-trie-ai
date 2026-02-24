from __future__ import annotations
import json
from pathlib import Path
import typer
from rich import print as rprint

from trie_ai.video_io import sample_frames_uniform, get_video_fps
from trie_ai.reason2_runner import Reason2Config, describe_frames_basic, generate_report
from trie_ai.report import now_iso
from trie_ai.prompts import load_prompts
from string import Template

app = typer.Typer(help="TRiE-AI: Traffic Incident Evidence Agent (CLI)")

@app.command()
def report(
    video: str = typer.Option(..., help="Path to input video (mp4)"),
    out: str = typer.Option(..., help="Path to output JSON report"),
    frames: int = typer.Option(8, help="Number of uniformly sampled frames"),
    workdir: str = typer.Option("results/tmp_frames", help="Where to store sampled frames"),
    backend: str = typer.Option("stub", help="Backend: stub | vllm_openai | vllm_inprocess"),
    model_size: str = typer.Option("2b", help="Model size selector: 2b | 8b"),
    model_id: str = typer.Option(None, help="Override model id, e.g. nvidia/Cosmos-Reason2-2B"),
    max_model_len: int = typer.Option(4096, help="Max model input length (vLLM in-process only)"),
    endpoint: str = typer.Option(None, help="vLLM endpoint host:port (or set env VLLM_ENDPOINT)"),
    api_key: str = typer.Option(None, help="vLLM API key (or set env VLLM_API_KEY)"),
    max_tokens: int = typer.Option(768, help="Max tokens"),
    temperature: float = typer.Option(0.2, help="Temperature"),
    prompts_path: str = typer.Option("configs/prompts.yaml", help="Prompts config path"),
):
    video_path = Path(video)
    if not video_path.exists():
        raise typer.BadParameter(f"Video not found: {video}")

    prompts = load_prompts(prompts_path)
    system_prompt = prompts["system"]
    # Keep your report prompt; it will be appended after the images.
    user_prompt = Template(prompts["report_prompt"]).safe_substitute(
        video_id=video_path.stem,
        fps="auto",
        frame_descriptions="(frames provided as images + indices/timestamps)"
    )

    fps = get_video_fps(str(video_path))
    samples = sample_frames_uniform(str(video_path), workdir, num_frames=frames)
    frame_tuples = [(s.index, s.timestamp_s, s.path) for s in samples]
    frame_desc = describe_frames_basic(frame_tuples)

    cfg = Reason2Config(
        mode=backend,
        model_size=model_size,
        model_id=model_id,
        endpoint=endpoint,
        api_key=api_key,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        temperature=temperature,
    )

    rep = generate_report(
        cfg,
        video_id=video_path.stem,
        fps=fps,
        frame_desc=frame_desc,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        video_path=str(video_path),
    )
    rep["metadata"]["generated_at"] = now_iso()

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    rprint("[bold green]OK[/bold green] Report written to:", str(out_path))
    rprint("Backend:", backend, "| Model:", rep["metadata"].get("model"))
    rprint("Sampled frames:", len(samples), "FPS:", fps)

def main():
    app()

if __name__ == "__main__":
    main()
