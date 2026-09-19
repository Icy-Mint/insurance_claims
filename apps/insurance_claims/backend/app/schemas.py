from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    phase: str
    ui_hints: Dict[str, Any] = {}
    state: Optional[Dict[str, Any]] = None
    # Driven directly off the backend SessionState object (SessionState.
    # as_debug_dict()), never separately reconstructed on the frontend —
    # this is what lets the UI's debug panel be trusted as ground truth.
    debug_state: Optional[Dict[str, Any]] = None
