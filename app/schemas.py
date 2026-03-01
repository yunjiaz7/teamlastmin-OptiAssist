from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: str


class UploadImageResponse(BaseModel):
    session_id: str
    image_url: str
    filename: str
    content_type: str
    size_bytes: int
    created_at: str


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    assistant_message: str
    confidence: str
    decision_path: dict[str, Any]
    tool_outputs: dict[str, Any]
    errors: list[str]
    history: list[ChatTurn]


class SessionResponse(BaseModel):
    session_id: str
    image_url: str
    filename: str
    content_type: str
    size_bytes: int
    created_at: str
    history: list[ChatTurn]
