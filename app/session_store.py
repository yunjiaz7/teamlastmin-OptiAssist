from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Literal
from uuid import uuid4

from app.schemas import ChatTurn


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionData:
    session_id: str
    image_path: str
    image_url: str
    filename: str
    content_type: str
    size_bytes: int
    created_at: str
    history: list[ChatTurn] = field(default_factory=list)


class InMemorySessionStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[str, SessionData] = {}

    def create_session(
        self,
        *,
        image_path: str,
        image_url: str,
        filename: str,
        content_type: str,
        size_bytes: int,
    ) -> SessionData:
        session_id = uuid4().hex
        session = SessionData(
            session_id=session_id,
            image_path=image_path,
            image_url=image_url,
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            created_at=utc_now_iso(),
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> SessionData | None:
        with self._lock:
            return self._sessions.get(session_id)

    def append_turn(self, session_id: str, role: Literal["user", "assistant"], content: str) -> SessionData | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.history.append(
                ChatTurn(
                    role=role,
                    content=content,
                    timestamp=utc_now_iso(),
                )
            )
            return session
