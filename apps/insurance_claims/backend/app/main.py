from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY from a local .env if present; never overrides an already-set env var

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .phases import process_turn
from .schemas import ChatRequest, ChatResponse
from .store import get_or_create_session, reset_session

app = FastAPI(title="SOP-Guided Insurance Claims Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    state = get_or_create_session(req.session_id)
    reply, public = process_turn(state, req.message)
    ui_hints = public.pop("ui_hints", {})
    return ChatResponse(
        reply=reply,
        phase=public["phase"],
        ui_hints=ui_hints,
        state=public,
        debug_state=state.as_debug_dict(),
    )


@app.post("/session/{session_id}/reset")
def reset(session_id: str) -> dict:
    reset_session(session_id)
    return {"status": "reset"}
