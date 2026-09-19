"""In-memory session store. No database needed for this demo."""
from __future__ import annotations

from typing import Dict

from .models import SessionState

_SESSIONS: Dict[str, SessionState] = {}


def get_or_create_session(session_id: str) -> SessionState:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = SessionState(session_id=session_id)
    return _SESSIONS[session_id]


def reset_session(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)


def reset_all() -> None:
    _SESSIONS.clear()
