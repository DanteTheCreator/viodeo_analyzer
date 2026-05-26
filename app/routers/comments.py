"""Timeline comment CRUD routes."""

import uuid

from fastapi import APIRouter, HTTPException

from app.models import Comment
from app.state import state

router = APIRouter()


@router.get("/comments")
async def list_comments():
    return {"comments": state["comments"]}


@router.post("/comments")
async def add_comment(comment: Comment):
    entry = {
        "id":        str(uuid.uuid4()),
        "timestamp": comment.timestamp,
        "text":      comment.text,
        "author":    comment.author,
    }
    state["comments"].append(entry)
    state["comments"].sort(key=lambda c: c["timestamp"])
    return entry


@router.delete("/comments/{comment_id}")
async def delete_comment(comment_id: str):
    before = len(state["comments"])
    state["comments"] = [c for c in state["comments"] if c["id"] != comment_id]
    if len(state["comments"]) == before:
        raise HTTPException(status_code=404, detail="Comment not found")
    return {"ok": True}
