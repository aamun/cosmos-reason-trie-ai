import json
import tempfile
from pathlib import Path
from string import Template
import streamlit as st

from trie_ai.video_io import sample_frames_uniform, get_video_fps
from trie_ai.reason2_runner import Reason2Config, describe_frames_basic, generate_report
from trie_ai.report import now_iso
from trie_ai.prompts import load_prompts

st.set_page_config(page_title="TRiE-AI", layout="wide")

st.title("Cosmos Reason 2 — Traffic Incident Evidence Agent (TRiE-AI)")
st.caption("Upload a traffic video → get an evidence-style JSON report (timeline + causal chain + risk).")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    uploaded = st.file_uploader("Upload MP4", type=["mp4", "mov", "m4v"])
    frames = st.slider("Sample frames (uniform)", min_value=4, max_value=20, value=8, step=1)
    mode = st.selectbox("Reason2 mode", ["stub", "vllm_inprocess", "vllm_openai"], index=0)

    run = st.button("Generate report", type="primary", disabled=(uploaded is None))

with col2:
    st.subheader("Output")
    out_box = st.empty()

if run and uploaded:
    with tempfile.TemporaryDirectory() as td:
        video_path = Path(td) / uploaded.name
        video_path.write_bytes(uploaded.getbuffer())

        fps = get_video_fps(str(video_path))
        frames_dir = Path(td) / "frames"
        samples = sample_frames_uniform(str(video_path), str(frames_dir), num_frames=frames)
        frame_tuples = [(s.index, s.timestamp_s, s.path) for s in samples]
        frame_desc = describe_frames_basic(frame_tuples)

        prompts = load_prompts("configs/prompts.yaml")
        system_prompt = prompts["system"]
        user_prompt = Template(prompts["report_prompt"]).safe_substitute(
            video_id=video_path.stem,
            fps=fps,
            frame_descriptions="(frames provided as images + indices/timestamps)",
        )

        cfg = Reason2Config(mode=mode)
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

        out_box.json(rep)

        st.subheader("Sampled frames")
        img_cols = st.columns(4)
        for i, s in enumerate(samples):
            with img_cols[i % 4]:
                st.image(str(s.path), caption=f"idx={s.index} t={s.timestamp_s:.2f}s", width='stretch')

        st.download_button(
            "Download JSON report",
            data=json.dumps(rep, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name=f"{video_path.stem}_report.json",
            mime="application/json",
        )
