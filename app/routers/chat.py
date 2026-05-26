"""Chat and AI-review routes."""

import uuid

from fastapi import APIRouter, HTTPException

from app.config import ORACLE_PASS_DEFS, load_instruction, load_prompt
from app.models import ChatRequest
from app.services.gemini import (
    chat_with_context,
    ensure_brand_logo,
    ensure_gemini_file,
    run_json_review,
)
from app.services.parser import extract_auto_comments
from app.state import state

router = APIRouter()


def _fmt_ts(seconds: float) -> str:
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m}:{s:05.2f}"


@router.post("/chat")
async def chat(req: ChatRequest):
    """Free-form chat: Gemini sees the video + current timeline and replies in prose."""
    if not state["video_path"]:
        raise HTTPException(status_code=400, detail="Upload a video first")

    chat_system_prompt = load_prompt("chat")

    if state["comments"]:
        lines = [
            f"[{_fmt_ts(c['timestamp'])}] ({c['author']}) {c['text']}"
            for c in state["comments"]
        ]
        comments_block = "\n\nExisting timeline comments:\n" + "\n".join(lines)
    else:
        comments_block = "\n\nNo timeline comments yet."

    # Always attach the video — it's already uploaded to Gemini File API.
    # Conditional attachment caused failures on follow-up messages like
    # "add these to the timeline" which have no visual trigger words.
    gemini_file_name = await ensure_gemini_file()

    answer = await chat_with_context(
        chat_system_prompt, req.message + comments_block,
        gemini_file_name=gemini_file_name,
        temperature=req.temperature,
    )
    # Opportunistically extract any COMMENT@ lines the model embedded in prose
    auto_comments = extract_auto_comments(answer)
    return {"reply": answer, "auto_comments": auto_comments}


@router.post("/chat/post-ai-comments")
async def post_ai_comments(req: ChatRequest):
    """Single-pass review: Gemini watches the whole video and returns JSON comments."""
    if not state["video_path"]:
        raise HTTPException(status_code=400, detail="Upload a video first")

    system_prompt    = load_prompt("professional")
    gemini_file_name = await ensure_gemini_file()
    brand_logo_name  = await ensure_brand_logo()

    brand_section = ""
    if brand_logo_name:
        brand_section = (
            "\n6. BRAND MATCH — an official brand reference logo image has been included in this "
            "request as a separate image file alongside the video.\n"
            "   THIS UPLOADED IMAGE IS THE ONLY AUTHORITATIVE VERSION OF THE LOGO. "
            "Do NOT treat any logo seen inside the video as the reference — the uploaded PNG is the ground truth.\n"
            "   Compare every logo, brand mark, label, and product visual in the video directly against "
            "the uploaded reference image.\n"
            "   Flag every mismatch: wrong shape, wrong icon, wrong colors, wrong font style, "
            "different proportions, distorted or placeholder logo.\n"
            "   Even if logos within the video look consistent with each other, "
            "they must still be compared against the uploaded reference — flag every one that differs.\n"
        )

    instruction = load_instruction("single.txt").replace("{brand_section}", brand_section)
    parsed = await run_json_review(
        system_prompt, gemini_file_name, instruction, brand_logo_name, "SINGLE PASS",
        temperature=req.temperature,
    )

    new_comments = []
    for ts, text in parsed:
        entry = {"id": str(uuid.uuid4()), "timestamp": ts, "text": text, "author": "ai"}
        state["comments"].append(entry)
        new_comments.append(entry)

    state["comments"].sort(key=lambda c: c["timestamp"])
    return {"ok": True, "added": new_comments}


@router.post("/chat/oracle")
async def oracle_mode(req: ChatRequest):
    """Multi-pass oracle review: 3 focused passes + optional brand pass."""
    if not state["video_path"]:
        raise HTTPException(status_code=400, detail="Upload a video first")

    system_prompt    = load_prompt("professional")
    gemini_file_name = await ensure_gemini_file()

    all_new      = []
    pass_results = []

    for pass_name, pass_file in ORACLE_PASS_DEFS:
        parsed = await run_json_review(
            system_prompt,
            gemini_file_name,
            load_instruction(pass_file),
            pass_label=f"ORACLE: {pass_name}",
            temperature=req.temperature,
        )
        for ts, text in parsed:
            entry = {"id": str(uuid.uuid4()), "timestamp": ts, "text": text, "author": "ai"}
            state["comments"].append(entry)
            all_new.append(entry)
        pass_results.append({"pass": pass_name, "count": len(parsed)})

    # Pass 4 — Brand Consistency (only when a logo is uploaded)
    brand_logo_name = await ensure_brand_logo()
    if brand_logo_name:
        parsed = await run_json_review(
            system_prompt,
            gemini_file_name,
            load_instruction("brand_pass.txt"),
            logo_file_name=brand_logo_name,
            pass_label="ORACLE: Pass 4 — Brand Consistency",
            temperature=req.temperature,
        )
        for ts, text in parsed:
            entry = {"id": str(uuid.uuid4()), "timestamp": ts, "text": text, "author": "ai"}
            state["comments"].append(entry)
            all_new.append(entry)
        pass_results.append({"pass": "Pass 4 — Brand Consistency", "count": len(parsed)})

    state["comments"].sort(key=lambda c: c["timestamp"])
    return {"added": all_new, "passes": pass_results}
