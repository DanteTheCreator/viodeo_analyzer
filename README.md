# Video Ad Reviewer

A FastAPI + Gemini-powered tool for reviewing AI-generated video ads. Upload a video, get timestamped comments on every AI artifact, quality issue, brand mismatch, and AV disconnect — directly on the video timeline.

## Features

- **Single Review** — one-pass scan of the full video; timestamped JSON comments posted to the timeline
- **The Oracle** — 3-pass deep review (Faces & Bodies → AI Text & Environments → AV, Audio & Image Quality) + optional Brand Consistency pass when a reference logo is uploaded
- **Brand Logo Check** — upload your brand logo PNG; the model compares every logo in the video against the reference and flags mismatches
- **Deterministic / Creative toggle** — T=0.2 for consistent, high-confidence comments; T=0.85 for wider net and more obscure catches
- **Chat** — free-form Q&A against the video and its current timeline (e.g. "what are the top 3 issues?", "what do you think about 0:08?"); model can post comments to the timeline on request
- **Click-to-seek** — clicking any comment jumps the video player to that exact timestamp

## Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12 · FastAPI 0.115.5 · Uvicorn |
| AI | Google Gemini (`gemini-3.5-flash`) via `google-generativeai` |
| Frontend | Single-file vanilla JS/HTML (`static/index.html`) |
| Video ingestion | Gemini File API (upload → ACTIVE → review → delete) |

## Setup

**1. Clone and create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Set your Gemini API key**
```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY=your_key_here
```

**3. Start the server**
```bash
uvicorn main:app --reload --port 8080
```

Open [http://localhost:8080](http://localhost:8080).

## Usage

1. **Upload a video** — drag and drop or click the upload area
2. *(Optional)* **Upload a brand logo** — PNG/JPG reference; activates brand consistency check in Oracle mode
3. **Choose a mode** — Deterministic (T=0.2) or Creative (T=0.85)
4. **Run a review** — click *Single Review* for a quick pass or *The Oracle* for a full multi-pass deep scan
5. **Interact** — type in the chat to ask questions or request specific comments be added to the timeline

## Project Structure

```
├── main.py                        # FastAPI app entry point
├── app/
│   ├── config.py                  # Model name, temperature, prompt paths
│   ├── models.py                  # Pydantic request/response models
│   ├── state.py                   # In-memory session state
│   ├── routers/
│   │   ├── chat.py                # /chat, /chat/oracle, /chat/post-ai-comments
│   │   ├── comments.py            # CRUD for timeline comments
│   │   └── upload.py              # Video + logo upload endpoints
│   └── services/
│       ├── gemini.py              # Gemini File API helpers, review runner
│       └── parser.py              # COMMENT@ extractor for free-form chat
├── prompts/
│   ├── professional.txt           # System prompt — Creative Strategist persona
│   ├── chat.txt                   # System prompt — video ad consultant chat persona
│   └── instructions/
│       ├── single.txt             # Single-pass review instruction (all 7 categories)
│       ├── oracle_pass1.txt       # Pass 1: Faces, bodies, AI skin artifacts
│       ├── oracle_pass2.txt       # Pass 2: AI text, environments, objects
│       ├── oracle_pass3.txt       # Pass 3: AV sync, image quality, audio, brand
│       └── brand_pass.txt         # Oracle Pass 4: brand logo vs reference image
├── static/
│   └── index.html                 # Full frontend (video player + timeline + chat)
├── tests/
│   ├── eval.py                    # Temperature comparison eval tool
│   └── comments.txt               # Ground truth comments for test videos
└── requirements.txt
```

## Eval Tool

`tests/eval.py` runs a single-pass review at multiple temperatures and produces a grouped comparison report.

```bash
# Run on all 3 test videos at default temps (0.2 and 0.85)
python tests/eval.py

# Custom temperatures
python tests/eval.py --temps 0.2,0.5,0.85,1.1

# Single video
python tests/eval.py --videos 1
```

Output is written to `tests/eval_temp_comparison.txt` — comments are grouped by timestamp cluster so you can see what each temperature found at the same moment.

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key (required) |

