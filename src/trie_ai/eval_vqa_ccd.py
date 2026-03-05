#!/usr/bin/env python3
"""
VQA Evaluation for CarCrashDataset (Crash-1500.txt) using TRiE AI JSON outputs.

Parses Crash-1500.txt and evaluates these VQA fields from your per-video JSON:
- q_day_night  -> {"Day","Night","Unknown"}
- q_weather    -> {"Normal","Rainy","Snowy","Unknown"}
- q_ego_involved -> {"Yes","No","Unknown"}

Computes:
- accuracy per field
- unknown_rate per field
- invalid_rate per field (answers outside the allowed choices)
- missing_file_rate and missing_field_rate

Assumptions:
- You save 1 JSON per video in --pred_dir
- Filenames contain the video_id (e.g., 001500.json or report_001500.json)
- JSON schema contains top-level keys: q_day_night, q_weather, q_ego_involved
- Ground truth mapping from Crash-1500.txt:
    timing: 0=Day, 1=Night
    weather: 0=Normal, 1=Snowy, 2=Rainy
    egoinvolve: 0=No, 1=Yes

Usage:
  python eval_vqa_ccd.py \
    --crash1500 /path/to/Crash-1500.txt \
    --pred_dir /path/to/outputs \
    --out results/vqa_metrics.json \
    --strict_mcq

Optional:
  --limit 200   (quick run)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DAY_NIGHT_CHOICES = {"Day", "Night", "Unknown"}
WEATHER_CHOICES = {"Normal", "Rainy", "Snowy", "Unknown"}
EGO_CHOICES = {"Yes", "No", "Unknown"}

FIELDS = ("q_day_night", "q_weather", "q_ego_involved")


def safe_load_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def parse_crash1500_line(line: str) -> Optional[Tuple[str, int, int, int, List[int]]]:
    """
    Expected line format:
      video_id,[binlabels...],event_start,event_end,Day|Night,Normal|Rainy|Snowy,Yes|No

    Example:
      000001,[0, 0, 0, ...],000285,0000,Day,Normal,Yes

    Returns:
      (video_id, timing, weather, egoinvolve, binlabels)
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    m = re.match(
        r"^\s*([^,]+)\s*,\s*(\[[^\]]*\])\s*,\s*[^,]*\s*,\s*[^,]*\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*$",
        line,
    )
    if not m:
        return None

    vidname = m.group(1).strip()
    binlabels_raw = m.group(2).strip()
    day_night = m.group(3).strip().lower()
    weather_str = m.group(4).strip().lower()
    ego_str = m.group(5).strip().lower()

    try:
        parsed_binlabels = json.loads(binlabels_raw)
    except Exception:
        return None
    if not isinstance(parsed_binlabels, list):
        return None

    binlabels: List[int] = []
    for tok in parsed_binlabels:
        try:
            binlabels.append(int(tok))
        except Exception:
            continue

    timing_map = {"day": 0, "night": 1}
    weather_map = {"normal": 0, "snowy": 1, "rainy": 2}
    ego_map = {"no": 0, "yes": 1}

    if day_night not in timing_map or weather_str not in weather_map or ego_str not in ego_map:
        return None

    timing = timing_map[day_night]
    weather = weather_map[weather_str]
    egoinvolve = ego_map[ego_str]

    return vidname, timing, weather, egoinvolve, binlabels


def load_crash1500(path: Path) -> Dict[str, dict]:
    gt: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            print(line.strip())
            parsed = parse_crash1500_line(line)
            if not parsed:
                continue
            vid, timing, weather, egoinvolve, binlabels = parsed
            gt[vid] = {
                "timing": timing,
                "weather": weather,
                "egoinvolve": egoinvolve,
                "binlabels": binlabels,
            }
    return gt


def gt_day_night(timing: int) -> str:
    return "Day" if timing == 0 else "Night"


def gt_weather(weather: int) -> str:
    if weather == 0:
        return "Normal"
    if weather == 1:
        return "Snowy"
    if weather == 2:
        return "Rainy"
    return "Unknown"


def gt_ego_involved(egoinvolve: int) -> str:
    return "Yes" if egoinvolve == 1 else "No"


def normalize_answer(ans: Optional[str]) -> str:
    if ans is None:
        return "Unknown"
    s = str(ans).strip()
    if not s:
        return "Unknown"

    low = s.lower()
    mapping = {
        # day/night
        "daytime": "Day",
        "nighttime": "Night",
        "day": "Day",
        "night": "Night",
        # weather
        "normal": "Normal",
        "clear": "Normal",
        "rain": "Rainy",
        "rainy": "Rainy",
        "snow": "Snowy",
        "snowy": "Snowy",
        # ego involved
        "yes": "Yes",
        "no": "No",
        # unknowns
        "unknown": "Unknown",
        "n/a": "Unknown",
        "na": "Unknown",
        "uncertain": "Unknown",
    }
    return mapping.get(low, s)


def find_pred_file(pred_dir: Path, video_id: str) -> Optional[Path]:
    """
    Prefer exact <video_id>.json.
    Otherwise find any *.json whose stem contains video_id as a numeric token.
    """
    exact = pred_dir / f"{video_id}.json"
    if exact.exists():
        return exact

    pattern = re.compile(rf"(?<!\d){re.escape(video_id)}(?!\d)")
    candidates = [p for p in pred_dir.glob("*.json") if pattern.search(p.stem)]
    if not candidates:
        return None

    candidates.sort(key=lambda p: (len(p.name), p.name))
    return candidates[0]


@dataclass
class FieldCounters:
    total: int = 0
    correct: int = 0
    unknown: int = 0
    invalid: int = 0
    missing_file: int = 0
    missing_field: int = 0


def evaluate(
    gt: Dict[str, dict],
    pred_dir: Path,
    strict_mcq: bool,
    limit: Optional[int] = None,
    debug_limit: int = 200,
) -> dict:
    counters: Dict[str, FieldCounters] = {f: FieldCounters() for f in FIELDS}
    debug: List[dict] = []

    video_ids = sorted(gt.keys())
    if limit is not None:
        video_ids = video_ids[:limit]

    for vid in video_ids:
        gt_answers = {
            "q_day_night": gt_day_night(gt[vid]["timing"]),
            "q_weather": gt_weather(gt[vid]["weather"]),
            "q_ego_involved": gt_ego_involved(gt[vid]["egoinvolve"]),
        }

        pred_path = find_pred_file(pred_dir, vid)
        if pred_path is None:
            for f in FIELDS:
                counters[f].total += 1
                counters[f].missing_file += 1
            if len(debug) < debug_limit:
                debug.append({"video_id": vid, "error": "missing_file"})
            continue

        pred = safe_load_json(pred_path)
        if pred is None:
            for f in FIELDS:
                counters[f].total += 1
                counters[f].missing_file += 1
            if len(debug) < debug_limit:
                debug.append({"video_id": vid, "error": "unreadable_json", "path": str(pred_path)})
            continue

        # Some users store video_id inside metadata; we don't rely on it.
        for f in FIELDS:
            counters[f].total += 1

            raw = pred.get(f, None)
            if raw is None:
                counters[f].missing_field += 1
                ans = "Unknown"
            else:
                ans = normalize_answer(raw)

            # validate against allowed choices
            allowed = DAY_NIGHT_CHOICES if f == "q_day_night" else WEATHER_CHOICES if f == "q_weather" else EGO_CHOICES
            if ans not in allowed:
                counters[f].invalid += 1
                if strict_mcq:
                    ans = "Unknown"

            if ans == "Unknown":
                counters[f].unknown += 1

            if ans == gt_answers[f]:
                counters[f].correct += 1
            else:
                if len(debug) < debug_limit:
                    debug.append({
                        "video_id": vid,
                        "field": f,
                        "gt": gt_answers[f],
                        "pred": ans,
                        "path": str(pred_path),
                    })

    # Build metrics
    per_field = {}
    macro_acc = 0.0
    macro_unknown = 0.0

    micro_total = 0
    micro_correct = 0
    micro_unknown = 0
    micro_missing_file = 0
    micro_missing_field = 0
    micro_invalid = 0

    for f in FIELDS:
        c = counters[f]
        denom = max(c.total, 1)

        acc = c.correct / denom
        unk = c.unknown / denom
        inv = c.invalid / denom
        mfile = c.missing_file / denom
        mfield = c.missing_field / denom

        per_field[f] = {
            "total": c.total,
            "correct": c.correct,
            "accuracy": round(acc, 6),
            "unknown": c.unknown,
            "unknown_rate": round(unk, 6),
            "invalid": c.invalid,
            "invalid_rate": round(inv, 6),
            "missing_file": c.missing_file,
            "missing_file_rate": round(mfile, 6),
            "missing_field": c.missing_field,
            "missing_field_rate": round(mfield, 6),
        }

        macro_acc += acc
        macro_unknown += unk

        micro_total += c.total
        micro_correct += c.correct
        micro_unknown += c.unknown
        micro_missing_file += c.missing_file
        micro_missing_field += c.missing_field
        micro_invalid += c.invalid

    macro_acc /= len(FIELDS)
    macro_unknown /= len(FIELDS)
    micro_acc = (micro_correct / micro_total) if micro_total else 0.0
    micro_unknown_rate = (micro_unknown / micro_total) if micro_total else 0.0

    results = {
        "overall": {
            "num_videos": len(video_ids),
            "fields_per_video": len(FIELDS),
            "macro_accuracy": round(macro_acc, 6),
            "micro_accuracy": round(micro_acc, 6),
            "macro_unknown_rate": round(macro_unknown, 6),
            "micro_unknown_rate": round(micro_unknown_rate, 6),
            "micro_missing_file": micro_missing_file,
            "micro_missing_field": micro_missing_field,
            "micro_invalid": micro_invalid,
            "strict_mcq": bool(strict_mcq),
        },
        "per_field": per_field,
        "debug_samples": debug[:debug_limit],
        "pred_dir": str(pred_dir),
    }
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crash1500", required=True, type=str, help="Path to Crash-1500.txt")
    ap.add_argument("--pred_dir", required=True, type=str, help="Directory with per-video prediction JSON files")
    ap.add_argument("--out", default="vqa_metrics.json", type=str, help="Where to write metrics JSON")
    ap.add_argument("--strict_mcq", action="store_true", help="Convert invalid answers to Unknown")
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only first N videos (for quick tests)")
    ap.add_argument("--debug_limit", type=int, default=200, help="Max mismatch samples to include in output")
    args = ap.parse_args()

    crash1500 = Path(args.crash1500)
    pred_dir = Path(args.pred_dir)
    out_path = Path(args.out)

    if not crash1500.exists():
        raise FileNotFoundError(f"Crash-1500.txt not found: {crash1500}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"pred_dir not found: {pred_dir}")

    gt = load_crash1500(crash1500)
    if not gt:
        raise RuntimeError("Parsed 0 entries from Crash-1500.txt")

    results = evaluate(
        gt=gt,
        pred_dir=pred_dir,
        strict_mcq=args.strict_mcq,
        limit=args.limit,
        debug_limit=args.debug_limit,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Console summary
    print("=== TRiE AI VQA Evaluation (CarCrashDataset Crash-1500) ===")
    print(f"Videos evaluated: {results['overall']['num_videos']}")
    print(f"Macro accuracy:   {results['overall']['macro_accuracy']}")
    print(f"Micro accuracy:   {results['overall']['micro_accuracy']}")
    print(f"Macro unknown:    {results['overall']['macro_unknown_rate']}")
    print(f"Micro unknown:    {results['overall']['micro_unknown_rate']}")
    print(f"Strict MCQ:       {results['overall']['strict_mcq']}")
    print(f"Wrote metrics to: {out_path}")


if __name__ == "__main__":
    main()
