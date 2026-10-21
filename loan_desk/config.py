"""Environment variables, API credentials, and lender policy limits.

Importing this module must never require network access or fail if LLM_API_KEY is
unset - tests import it without a .env file. Callers that actually need the key (the
real LLM client, or the server entry point) check for it themselves and fail there.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

MAX_STEPS = 6  # agent loop iteration guard
MAX_IMAGES = 3
MAX_BODY_BYTES = 8 * 1024 * 1024  # a few JPEG data URLs plus text, generously bounded


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ, without overwriting existing vars."""
    for line in path.read_text().splitlines() if path.exists() else []:
        m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*?)\s*$", line)
        if m:
            os.environ.setdefault(m[1], m[2])


_load_dotenv(HERE / ".env")

KEY = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
BASE = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
MODEL = os.environ.get("MODEL", "gemini-flash-lite-latest")

# EDIT ME: example limits only, not regulations. Set your lender's real policy in policy.json.
DEFAULT_POLICY = {"max_pd": 0.15, "max_pti": 0.20, "max_dti_after": 0.43, "max_ltv": 0.80,
                   "min_income": 24000, "min_fico": 600, "default_rate_pct": 12.0,
                   "currency": "$", "usd_per_unit": 1.0}


def load_policy(path: Path) -> dict:
    """Load lender policy limits from policy.json, creating it with defaults on first run."""
    policy = dict(DEFAULT_POLICY)
    if path.exists():
        policy.update(json.loads(path.read_text()))
    else:
        path.write_text(json.dumps(policy, indent=2))
    return policy


POLICY = load_policy(HERE / "policy.json")
POLICY_DOC_PATH = HERE / "policy_docs" / "lender_policy.md"
