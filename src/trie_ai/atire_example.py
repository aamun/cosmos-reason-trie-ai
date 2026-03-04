from trie_ai.atire import Atire, AtireConfig
from trie_ai.prompts import load_prompts

# Load prompts and prepare frame descriptions
prompts = load_prompts("configs/prompts.yaml")
system_prompt = prompts["system"]
# Keep your report prompt; it will be appended after the images.
user_prompt = prompts["report_prompt"]

cfg = AtireConfig(
    mode="stub", 
    model_path="nvidia/Cosmos-Reason2-2B",
    nframes=8,
)
with Atire(cfg) as reasoner:
    text = reasoner.generate_report(system_prompt=system_prompt, user_prompt=user_prompt, video_path="data/demo_clips/sample.mp4")
    print(text)
    text = reasoner.generate_report(system_prompt=system_prompt, user_prompt=user_prompt, video_path="data/CarCrash/Crash-1500/000001.mp4")
    print(text)