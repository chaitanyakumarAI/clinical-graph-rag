"""Shared test setup: project root on sys.path, no real LLM keys, an LLM fake."""

from __future__ import annotations
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_LLM_ENV_VARS = ["LLM_PROVIDER", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"]


@pytest.fixture(autouse=True)
def _no_real_llm_keys(monkeypatch):
    """Make sure no test can reach a real provider with a developer's key."""
    for name in _LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class FakeLLM:
    """
    Stand-in for llm.chat / llm.chat_json.

    `replies` is a list of return values (or exceptions to raise), consumed in
    order. Every call's (system, user, kwargs) is recorded in `calls`.
    """

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, system, user, **kwargs):
        self.calls.append({"system": system, "user": user, **kwargs})
        if not self.replies:
            raise AssertionError("FakeLLM called more times than expected")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


@pytest.fixture
def fake_llm():
    return FakeLLM
