from __future__ import annotations
from pathlib import Path
import yaml

def load_prompts(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))
