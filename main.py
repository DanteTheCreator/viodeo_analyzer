import os
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app.routers import chat, comments, upload

load_dotenv()

_api_key = os.getenv("GEMINI_API_KEY", "")
if _api_key:
    genai.configure(api_key=_api_key)

app = FastAPI(title="Video Ad Reviewer")

app.include_router(upload.router)
app.include_router(comments.router)
app.include_router(chat.router)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path("static/index.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(html_path.read_text())
