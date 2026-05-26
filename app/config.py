import os
from pathlib import Path

# ── Model / generation ────────────────────────────────────────────────────────
MODEL       = "gemini-3.5-flash"
TEMPERATURE = 2.0  # lower = more focused; range 0.0–2.0

# ── Paths ─────────────────────────────────────────────────────────────────────
PROMPTS_DIR      = Path("prompts")
INSTRUCTIONS_DIR = Path("prompts/instructions")
UPLOAD_DIR       = Path("uploads")

PROMPT_FILES = {
    "professional": PROMPTS_DIR / "professional.txt",
    "nitpick":      PROMPTS_DIR / "nitpick.txt",
    "chat":         PROMPTS_DIR / "chat.txt",
}

ORACLE_PASS_DEFS = [
    ("Pass 1 — Faces & Bodies",                        "oracle_pass1.txt"),
    ("Pass 2 — AI Text & Environments",                "oracle_pass2.txt"),
    ("Pass 3 — AV, Image Quality & Audio",             "oracle_pass3.txt"),
]

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_prompt(mode: str) -> str:
    path = PROMPT_FILES.get(mode, PROMPT_FILES["professional"])
    return path.read_text()


def load_instruction(filename: str) -> str:
    return (INSTRUCTIONS_DIR / filename).read_text()


def gemini_api_key() -> str:
    return os.getenv("GEMINI_API_KEY", "")
