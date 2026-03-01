from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.agent.langgraph_agent import analyze_fundus_case
from app.schemas import ChatRequest, ChatResponse, SessionResponse, UploadImageResponse
from app.session_store import InMemorySessionStore
from app.upload_service import save_upload_file

ROOT_DIR = Path(__file__).resolve().parents[1]
UPLOADS_DIR = ROOT_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="TeamLastMin Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

SESSION_STORE = InMemorySessionStore()


def _build_contextual_query(current_message: str, history: list[dict[str, str]]) -> str:
    if not history:
        return current_message
    recent_turns = history[-6:]
    context_lines = [f"{turn['role']}: {turn['content']}" for turn in recent_turns]
    context_block = "\n".join(context_lines)
    return (
        f"{current_message}\n\n"
        "Conversation context for the same fundus image:\n"
        f"{context_block}"
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/upload-image", response_model=UploadImageResponse)
async def upload_image(file: UploadFile = File(...)) -> UploadImageResponse:
    session_upload_dir = UPLOADS_DIR / "tmp"
    saved = await save_upload_file(upload=file, destination_dir=session_upload_dir)
    session = SESSION_STORE.create_session(
        image_path=str(saved.path),
        image_url=f"/uploads/{saved.path.relative_to(UPLOADS_DIR).as_posix()}",
        filename=saved.filename,
        content_type=saved.content_type,
        size_bytes=saved.size_bytes,
    )
    final_dir = UPLOADS_DIR / session.session_id
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / saved.filename
    saved.path.rename(final_path)
    session.image_path = str(final_path)
    session.image_url = f"/uploads/{session.session_id}/{saved.filename}"
    return UploadImageResponse(
        session_id=session.session_id,
        image_url=session.image_url,
        filename=session.filename,
        content_type=session.content_type,
        size_bytes=session.size_bytes,
        created_at=session.created_at,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    session = SESSION_STORE.get_session(payload.session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown session_id: {payload.session_id}",
        )

    SESSION_STORE.append_turn(session.session_id, "user", payload.message)
    updated_session = SESSION_STORE.get_session(session.session_id)
    if updated_session is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Session lost unexpectedly.")

    history_dict = [{"role": turn.role, "content": turn.content} for turn in updated_session.history[:-1]]
    contextual_query = _build_contextual_query(payload.message, history_dict)

    try:
        agent_result = analyze_fundus_case(query=contextual_query, image_path=updated_session.image_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {exc}",
        ) from exc

    assistant_message = str(agent_result.get("final_answer", "")).strip()
    SESSION_STORE.append_turn(session.session_id, "assistant", assistant_message)
    refreshed = SESSION_STORE.get_session(session.session_id)
    if refreshed is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Session lost unexpectedly.")

    return ChatResponse(
        session_id=refreshed.session_id,
        assistant_message=assistant_message,
        confidence=str(agent_result.get("confidence", "low")),
        decision_path=dict(agent_result.get("decision_path", {})),
        tool_outputs=dict(agent_result.get("tool_outputs", {})),
        errors=[str(e) for e in agent_result.get("errors", [])],
        history=refreshed.history,
    )


@app.get("/api/session/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    session = SESSION_STORE.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown session_id: {session_id}",
        )
    return SessionResponse(
        session_id=session.session_id,
        image_url=session.image_url,
        filename=session.filename,
        content_type=session.content_type,
        size_bytes=session.size_bytes,
        created_at=session.created_at,
        history=session.history,
    )
