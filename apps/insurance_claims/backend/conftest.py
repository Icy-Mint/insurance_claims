import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import app.main  # noqa: E402  (imported here so its load_dotenv() runs before we strip the key below)

# Tests exercise the deterministic fallback path so they run without network
# access or a real API key, regardless of whether a local .env sets one, or
# whether one is injected via a CI secret's environment variable. Stashed
# under a different name (rather than just discarded) so the `real_llm`
# fixture in tests/conftest.py can still recover it for the handful of tests
# that specifically want genuine live-model behavior — this makes those
# tests work the same way locally (via backend/.env) and in CI (via an env
# var / repo secret), instead of only supporting one or the other.
os.environ["_STASHED_ANTHROPIC_API_KEY"] = os.environ.pop("ANTHROPIC_API_KEY", "")
