from pydantic import BaseModel


class Comment(BaseModel):
    timestamp: float
    text: str
    author: str = "user"  # "user" | "ai"


class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7


class CommentItem(BaseModel):
    """JSON schema shape that Gemini is constrained to output for review passes.

    Pydantic BaseModel is required here — google-generativeai converts it to
    the proto Schema format (type_: ARRAY / OBJECT / NUMBER / STRING).
    TypedDict is NOT converted and the schema constraint is silently ignored.
    """
    timestamp: float   # seconds from video start
    comment: str
