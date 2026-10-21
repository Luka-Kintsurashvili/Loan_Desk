"""Thin client for an OpenAI-compatible chat-completions endpoint, including function calling."""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from . import config

logger = logging.getLogger(__name__)


def call_llm(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Make one chat-completions call and return the assistant message (may include tool_calls).

    Raises RuntimeError if no API key is configured or the endpoint returns an HTTP error.
    """
    if not config.KEY:
        raise RuntimeError("Put LLM_API_KEY=... in the .env file")
    payload = {"model": config.MODEL, "messages": messages}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        payload["reasoning_effort"] = "none"  # this endpoint rejects function tools otherwise
    req = urllib.request.Request(
        config.BASE + "/chat/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + config.KEY})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)["choices"][0]["message"]
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        logger.warning("LLM call failed: HTTP %s", e.code)  # never log the body: it echoes our request
        raise RuntimeError(f"LLM {e.code}: {body}")
