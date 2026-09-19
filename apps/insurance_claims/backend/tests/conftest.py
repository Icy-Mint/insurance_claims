import os
import uuid
from pathlib import Path

import pytest
from dotenv import dotenv_values
from fastapi.testclient import TestClient

from app.main import app
from app.store import reset_all
from app import llm as llm_module
from app import tools as tools_module

# The top-level backend/conftest.py strips ANTHROPIC_API_KEY from the
# environment for the whole test session, so every test runs against the
# deterministic fallback path by default (fast, offline, no flakiness). A
# handful of tests specifically want to exercise genuine live-model language
# understanding (messy phrasing, compound intents) — the `real_llm` fixture
# below is how they opt into that, without affecting any other test.
_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


@pytest.fixture(autouse=True)
def _clean_state():
    reset_all()
    tools_module.EMAIL_LOG.clear()
    yield
    reset_all()
    tools_module.EMAIL_LOG.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def session_id():
    return f"test-{uuid.uuid4()}"


@pytest.fixture
def real_llm():
    """Temporarily restores a real ANTHROPIC_API_KEY (read directly from
    backend/.env, bypassing the global strip) and resets llm.py's cached
    client so the test hits the actual Anthropic API. Restores the offline
    state afterward so later tests aren't affected. Skips (rather than
    failing) if no key is configured, so the suite stays runnable in CI
    environments without one."""
    # Checks, in order: an already-set env var (shouldn't normally happen —
    # the top-level conftest strips it), the value the top-level conftest
    # stashed before stripping (covers a CI secret passed as an env var),
    # and finally a local backend/.env file directly (covers local dev).
    key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("_STASHED_ANTHROPIC_API_KEY")
        or dotenv_values(_ENV_PATH).get("ANTHROPIC_API_KEY")
    )
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not configured (checked env and backend/.env) — skipping live-API test")

    old_env = os.environ.get("ANTHROPIC_API_KEY")
    old_client, old_checked = llm_module._client, llm_module._client_checked

    os.environ["ANTHROPIC_API_KEY"] = key
    llm_module._client = None
    llm_module._client_checked = False
    try:
        client = llm_module.get_client()
        if client is None:
            pytest.skip("Anthropic client failed to initialize with the configured key")
        yield
    finally:
        if old_env is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = old_env
        llm_module._client = old_client
        llm_module._client_checked = old_checked
