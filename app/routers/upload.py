"""Upload and video-streaming routes."""

import uuid
from pathlib import Path

import aiofiles
import google.generativeai as genai
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import UPLOAD_DIR
from app.state import state

router = APIRouter()

UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Save uploaded video locally. Gemini upload is deferred to first analysis."""
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    ext = Path(file.filename).suffix or ".mp4"
    local_path = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"

    async with aiofiles.open(local_path, "wb") as f:
        await f.write(await file.read())

    state.update({
        "video_path":       str(local_path),
        "video_mime":       file.content_type,
        "gemini_file_uri":  None,
        "gemini_file_name": None,
        "comments":         [],
        "chat_history":     [],
    })
    return {"ok": True, "filename": file.filename}


@router.post("/upload-logo")
async def upload_logo(file: UploadFile = File(...)):
    """Save brand reference logo locally for brand consistency checks."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    ext = Path(file.filename).suffix or ".png"
    local_path = UPLOAD_DIR / f"logo_{uuid.uuid4()}{ext}"

    async with aiofiles.open(local_path, "wb") as f:
        await f.write(await file.read())

    # Invalidate any cached Gemini logo file
    old_name = state.get("brand_logo_gemini_name")
    if old_name:
        try:
            genai.delete_file(old_name)
        except Exception:
            pass

    state.update({
        "brand_logo_path":        str(local_path),
        "brand_logo_mime":        file.content_type,
        "brand_logo_gemini_name": None,
    })
    return {"ok": True, "filename": file.filename}


@router.get("/video")
async def stream_video():
    """Serve the current video for the browser <video> element."""
    if not state["video_path"]:
        raise HTTPException(status_code=404, detail="No video uploaded yet")
    return FileResponse(state["video_path"], media_type="video/mp4")
