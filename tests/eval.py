#!/usr/bin/env python3
"""
Temperature Comparison Eval — single-pass review at multiple temperatures.

Runs each video at every temperature, then writes one file that groups
comments by timestamp so you can see what each temperature found at the
same moment in the video.

Usage:
    python tests/eval.py
    python tests/eval.py --temps 0.2,0.85
    python tests/eval.py --videos 1,2 --temps 0.2,0.85
    python tests/eval.py --output tests/my_run.txt
    python tests/eval.py --mode professional --videos 1,2,3
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(Path(__file__).parent.parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not set in .env")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

TESTS_DIR        = Path(__file__).parent
PROMPTS_DIR      = Path(__file__).parent.parent / "prompts"
INSTRUCTIONS_DIR = PROMPTS_DIR / "instructions"

MODEL            = "gemini-3.5-flash"
DEFAULT_TEMPS    = [0.2, 0.85]
CLUSTER_WINDOW_S = 3.0   # comments within this range are grouped as the same "moment"
OUTPUT_WIDTH     = 110   # characters wide for the output file


# ── Gemini helpers ─────────────────────────────────────────────────────────

def upload_video(video_path: Path) -> str:
    """Delete all existing Gemini files then upload the new one. Returns file name."""
    deleted = 0
    try:
        for old in genai.list_files():
            try:
                genai.delete_file(old.name)
                deleted += 1
            except Exception:
                pass
    except Exception:
        pass
    if deleted:
        print(f"    purged {deleted} old Gemini file(s)")

    for attempt in range(1, 4):
        gf = genai.upload_file(path=str(video_path), mime_type="video/mp4")
        print(f"    upload attempt {attempt}: {gf.name}", end="", flush=True)
        while gf.state.name == "PROCESSING":
            time.sleep(2)
            gf = genai.get_file(gf.name)
            print(".", end="", flush=True)
        if gf.state.name == "ACTIVE":
            print(" ACTIVE")
            return gf.name
        print(f" FAILED — retrying in 3s...")
        try:
            genai.delete_file(gf.name)
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("Video failed to reach ACTIVE state after 3 attempts")


def _parse_json_comments(raw: str) -> list[tuple[float, str]]:
    """Robust parser: strips markdown fences, finds the first [...] block."""
    text = re.sub(r"```[\w]*\n?", "", raw).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
        result = []
        for item in items:
            if "timestamp" in item and "comment" in item:
                ts = float(item["timestamp"])
                if 0.0 <= ts <= 3600.0:
                    result.append((round(ts, 2), str(item["comment"])))
        return result
    except Exception:
        return []


def run_single_review(gemini_file_name: str, system_prompt: str, temperature: float) -> list[tuple[float, str]]:
    """Run one single-pass review at the given temperature."""
    instruction = (INSTRUCTIONS_DIR / "single.txt").read_text().replace("{brand_section}", "")
    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=system_prompt,
        generation_config=genai.types.GenerationConfig(temperature=temperature),
    )
    video_file = genai.get_file(gemini_file_name)
    resp = model.generate_content([video_file, instruction])
    raw = resp.text if resp.parts else "[]"
    parsed = _parse_json_comments(raw)

    # Retry once if response looks like it had content but failed to parse
    if not parsed and raw.strip() not in ("[]", "[ ]", ""):
        resp = model.generate_content([video_file, instruction])
        raw = resp.text if resp.parts else "[]"
        parsed = _parse_json_comments(raw)

    return parsed


# ── Clustering ─────────────────────────────────────────────────────────────

def cluster_comments(
    by_temp: dict,
    window: float,
) -> list:
    """
    Group comments from all temperatures into shared "moments" by timestamp proximity.
    Returns list of {"center_ts": float, "comments": {temp: str|None}}
    """
    flat = []  # (ts, temp, comment)
    for temp, comments in by_temp.items():
        for ts, comment in comments:
            flat.append((ts, temp, comment))

    if not flat:
        return []

    flat.sort(key=lambda x: x[0])

    # Greedy single-pass clustering
    clusters = []
    for ts, temp, comment in flat:
        placed = False
        for c in clusters:
            if abs(ts - c["center_ts"]) <= window:
                c["items"].append((ts, temp, comment))
                c["center_ts"] = sum(t for t, _, _ in c["items"]) / len(c["items"])
                placed = True
                break
        if not placed:
            clusters.append({"center_ts": ts, "items": [(ts, temp, comment)]})

    all_temps = sorted(by_temp.keys())
    result = []
    for c in sorted(clusters, key=lambda x: x["center_ts"]):
        temp_comments = {}
        for temp in all_temps:
            matches = [(ts, cmt) for ts, t, cmt in c["items"] if t == temp]
            if matches:
                best = min(matches, key=lambda x: abs(x[0] - c["center_ts"]))
                temp_comments[temp] = best[1]
            else:
                temp_comments[temp] = None
        result.append({"center_ts": c["center_ts"], "comments": temp_comments})

    return result


# ── Output formatting ──────────────────────────────────────────────────────

def _fmt_temp(t: float) -> str:
    return f"T={t:.2f}"


def _wrap_text(text: str, indent: str, first_prefix: str, max_width: int) -> list:
    """Word-wrap text into lines. First line uses first_prefix, rest use indent."""
    words = text.split()
    lines = []
    current = ""
    prefix = first_prefix
    for word in words:
        if current and len(current) + 1 + len(word) > max_width - len(prefix):
            lines.append(f"{prefix}{current}")
            prefix = indent
            current = word
        else:
            current = (current + " " + word).lstrip()
    if current:
        lines.append(f"{prefix}{current}")
    return lines


def write_video_section(fh, video_num: int, temps: list, by_temp: dict) -> None:
    W = OUTPUT_WIDTH
    label_w = max(len(_fmt_temp(t)) for t in temps)
    tag_w   = label_w + 6  # "  T=1.50 │ "
    text_w  = W - tag_w

    fh.write(f"\n{'═' * W}\n")
    fh.write(f"  VIDEO {video_num}\n")
    fh.write(f"{'═' * W}\n")

    clusters = cluster_comments(by_temp, CLUSTER_WINDOW_S)

    if not clusters:
        fh.write("\n  (no comments returned at any temperature)\n")
    else:
        for cluster in clusters:
            fh.write(f"\n  ~{cluster['center_ts']:.1f}s\n")
            for temp in temps:
                comment = cluster["comments"].get(temp)
                tag     = f"  {_fmt_temp(temp):<{label_w}} │ "
                indent  = " " * len(tag)
                if comment:
                    for line in _wrap_text(comment, indent, tag, W):
                        fh.write(line + "\n")
                else:
                    fh.write(f"{tag}—\n")
            fh.write(f"  {'·' * (W - 2)}\n")

    fh.write(f"\n  Comments returned per temperature:\n")
    for temp in temps:
        count = len(by_temp.get(temp, []))
        bar   = "█" * count
        fh.write(f"    {_fmt_temp(temp):<8}  {count:>3}  {bar}\n")
    fh.write("\n")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Temperature Comparison Eval — single-pass")
    parser.add_argument(
        "--temps", default=",".join(str(t) for t in DEFAULT_TEMPS),
        help=f"Comma-separated temperatures to test (default: {','.join(str(t) for t in DEFAULT_TEMPS)})",
    )
    parser.add_argument(
        "--videos", default="1,2,3",
        help="Comma-separated video numbers (default: 1,2,3)",
    )
    parser.add_argument(
        "--output", default=str(TESTS_DIR / "eval_temp_comparison.txt"),
        help="Output file path",
    )
    parser.add_argument(
        "--mode", default="professional",
        help="System prompt to use — must match a file in prompts/ (default: professional)",
    )
    args = parser.parse_args()

    temps         = [float(t.strip()) for t in args.temps.split(",")]
    video_nums    = [int(v.strip())   for v in args.videos.split(",")]
    system_prompt = (PROMPTS_DIR / f"{args.mode}.txt").read_text()
    out_path      = Path(args.output)

    W = 60
    print(f"\n{'=' * W}")
    print(f"  Temperature Comparison Eval")
    print(f"  Mode   : {args.mode}")
    print(f"  Videos : {video_nums}")
    print(f"  Temps  : {temps}")
    print(f"  Output : {out_path.name}")
    print(f"{'=' * W}\n")

    grand_totals = {t: 0 for t in temps}

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Temperature Comparison Eval\n")
        fh.write(f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"Mode      : {args.mode}\n")
        fh.write(f"Videos    : {video_nums}\n")
        fh.write(f"Temps     : {temps}\n")
        fh.write(f"Cluster   : comments within +/-{CLUSTER_WINDOW_S}s grouped as same moment\n")

        for video_num in video_nums:
            video_path = TESTS_DIR / f"{video_num}.mp4"
            if not video_path.exists():
                print(f"  SKIP: {video_path.name} not found")
                continue

            print(f"  -- VIDEO {video_num} --")
            print(f"  Uploading...")
            try:
                gemini_name = upload_video(video_path)
            except RuntimeError as e:
                print(f"  ERROR: {e}")
                continue

            by_temp = {}

            for temp in temps:
                print(f"  {_fmt_temp(temp)} running...", end="", flush=True)
                try:
                    comments = run_single_review(gemini_name, system_prompt, temp)
                except Exception as e:
                    print(f" ERROR: {e}")
                    comments = []
                by_temp[temp] = comments
                grand_totals[temp] += len(comments)
                print(f" -> {len(comments)} comment(s)")

            write_video_section(fh, video_num, temps, by_temp)
            fh.flush()

            try:
                genai.delete_file(gemini_name)
            except Exception:
                pass
            print()

        # Grand summary
        fh.write(f"\n{'=' * OUTPUT_WIDTH}\n")
        fh.write(f"  GRAND TOTALS  (all videos combined)\n")
        fh.write(f"{'=' * OUTPUT_WIDTH}\n\n")
        for temp in temps:
            count = grand_totals[temp]
            bar   = "█" * count
            fh.write(f"  {_fmt_temp(temp):<8}  {count:>3}  {bar}\n")
        fh.write("\n")

    print(f"  Output written to: {out_path}\n")


if __name__ == "__main__":
    main()
