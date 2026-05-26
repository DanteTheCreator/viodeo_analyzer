"""Gemini API helpers — file management, structured review passes, and free-form chat."""

import asyncio
import json
import time
from typing import Optional

import google.generativeai as genai
from fastapi import HTTPException

from app.config import MODEL, TEMPERATURE, gemini_api_key
from app.models import CommentItem
from app.state import state


# ── Model factory ─────────────────────────────────────────────────────────────

def _build_model(system_instruction: str, *, temperature: float = TEMPERATURE) -> genai.GenerativeModel:
    """Return a configured GenerativeModel."""
    config: dict = {"temperature": temperature}
    return genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=system_instruction,
        generation_config=genai.types.GenerationConfig(**config),
    )


# ── JSON response parser ───────────────────────────────────────────────────────

def parse_json_comments(raw: str) -> list[tuple[float, str]]:
    """Parse a JSON array from a raw model response.

    Handles responses wrapped in markdown code fences or surrounded by prose
    by extracting the first [...] block found in the text.
    """
    import re
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r'```[\w]*\n?', '', raw).strip()
    # Find the first JSON array in the text
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
        return [
            (float(item["timestamp"]), str(item["comment"]))
            for item in items
            if "timestamp" in item and "comment" in item
        ]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return []


# ── Gemini File API helpers ────────────────────────────────────────────────────

def _raise_network_or_500(exc: Exception, context: str) -> None:
    """Raise 503 for DNS/network failures, 500 for anything else."""
    msg = str(exc)
    if "ServerNotFoundError" in type(exc).__name__ or "Unable to find the server" in msg or "nodename nor servname" in msg:
        raise HTTPException(
            status_code=503,
            detail=f"{context}: cannot reach Gemini API — check network connection",
        )
    raise HTTPException(status_code=500, detail=f"{context}: {msg}")


async def ensure_gemini_file() -> str:
    """Upload the current video to Gemini File API on first call; reuse on subsequent calls."""
    if not gemini_api_key():
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
    if state["gemini_file_name"]:
        return state["gemini_file_name"]
    if not state["video_path"]:
        raise HTTPException(status_code=400, detail="Upload a video first")

    local_path = state["video_path"]
    mime = state.get("video_mime", "video/mp4")

    def _upload():
        deleted = 0
        logo_name = state.get("brand_logo_gemini_name")
        try:
            for old_gf in genai.list_files():
                if logo_name and old_gf.name == logo_name:
                    continue
                try:
                    genai.delete_file(old_gf.name)
                    deleted += 1
                except Exception:
                    pass
        except Exception:
            pass
        print(f"[upload] purged {deleted} old Gemini file(s)")

        for attempt in range(1, 4):
            gf = genai.upload_file(path=local_path, mime_type=mime)
            print(f"[upload] attempt {attempt}: {gf.name} — waiting for ACTIVE...")
            while gf.state.name == "PROCESSING":
                time.sleep(2)
                gf = genai.get_file(gf.name)
            if gf.state.name == "ACTIVE":
                print(f"[upload] {gf.name} is ACTIVE")
                return gf
            print(f"[upload] attempt {attempt} ended in {gf.state.name}, retrying...")
            try:
                genai.delete_file(gf.name)
            except Exception:
                pass
            time.sleep(3)
        raise RuntimeError("Gemini file failed to reach ACTIVE state after 3 attempts")

    try:
        gf = await asyncio.to_thread(_upload)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        _raise_network_or_500(e, "Video upload failed")

    state["gemini_file_uri"] = gf.uri
    state["gemini_file_name"] = gf.name
    return gf.name


async def ensure_brand_logo() -> Optional[str]:
    """Upload the brand logo to Gemini File API on first call; reuse on subsequent calls."""
    if not state["brand_logo_path"]:
        return None
    if state["brand_logo_gemini_name"]:
        return state["brand_logo_gemini_name"]

    def _upload_logo():
        gf = genai.upload_file(
            path=state["brand_logo_path"],
            mime_type=state["brand_logo_mime"],
        )
        while gf.state.name == "PROCESSING":
            time.sleep(1)
            gf = genai.get_file(gf.name)
        if gf.state.name != "ACTIVE":
            raise RuntimeError(f"Brand logo failed to reach ACTIVE: {gf.state.name}")
        print(f"[logo] {gf.name} is ACTIVE")
        return gf

    try:
        gf = await asyncio.to_thread(_upload_logo)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Brand logo upload: {e}")
    except Exception as e:
        _raise_network_or_500(e, "Brand logo upload failed")

    state["brand_logo_gemini_name"] = gf.name
    return gf.name


# ── Review helpers ─────────────────────────────────────────────────────────────

async def run_json_review(
    system_prompt: str,
    gemini_file_name: str,
    instruction: str,
    logo_file_name: Optional[str] = None,
    pass_label: Optional[str] = None,
    temperature: float = TEMPERATURE,
) -> list[tuple[float, str]]:
    """Run one Gemini review pass with JSON output enforced.

    Returns a list of (timestamp_seconds, comment_text) pairs.
    The response_schema constrains Gemini to emit a typed JSON array, so
    timestamps are always numbers — no string parsing or M.SS ambiguity.
    """
    def _call():
        model = _build_model(system_prompt, temperature=temperature)
        parts = [genai.get_file(gemini_file_name)]
        if logo_file_name:
            parts.append(genai.get_file(logo_file_name))
        parts.append(instruction)
        resp = model.generate_content(parts)
        return resp.text if resp.parts else "[]"

    try:
        raw = await asyncio.to_thread(_call)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if pass_label:
        print(f"\n===== {pass_label} =====")
        print(raw)
        print("===== END =====\n")

    parsed = parse_json_comments(raw)

    if pass_label:
        print(f"  → {len(parsed)} comments parsed\n")

    return parsed


async def chat_with_context(
    system_prompt: str,
    user_content: str,
    *,
    gemini_file_name: Optional[str] = None,
    temperature: float = TEMPERATURE,
) -> str:
    """Free-form chat — no JSON schema constraint, returns raw model text.

    Pass gemini_file_name to attach the video so Gemini can inspect frames.
    Omit it (or pass None) for text-only replies — no upload, instant response.
    """
    def _call():
        model = _build_model(system_prompt, temperature=temperature)
        if gemini_file_name:
            parts = [genai.get_file(gemini_file_name), user_content]
        else:
            parts = [user_content]
        response = model.generate_content(parts)
        return response.text if response.parts else ""

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
