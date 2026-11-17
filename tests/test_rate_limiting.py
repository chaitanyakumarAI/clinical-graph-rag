"""llm/rate_limiter.py and the 429 handling in llm/client.py."""

from types import SimpleNamespace

import httpx
import openai
import pytest

from llm import client as llm_client
from llm import rate_limiter
from llm.client import RateLimitExhausted, _is_daily_quota, _retry_after_seconds
from llm.rate_limiter import RateLimiter


class FakeClock:
    """Replaces time.monotonic / time.sleep so tests run instantly."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    # Replace the module's `time` reference, not the global time module.
    monkeypatch.setattr(rate_limiter, "time", SimpleNamespace(monotonic=fake.monotonic, sleep=fake.sleep))
    return fake


# ── RateLimiter ───────────────────────────────────────────────────────────────

def test_requests_per_minute(clock):
    limiter = RateLimiter("test", rpm=2)
    limiter.acquire(10)
    clock.now += 5
    limiter.acquire(10)
    assert clock.sleeps == []

    limiter.acquire(10)   # third request must wait until the first one is 60 s old
    assert sum(clock.sleeps) == pytest.approx(55.0)


def test_tokens_per_minute(clock):
    limiter = RateLimiter("test", tpm=1000)
    limiter.acquire(600)
    clock.now += 10
    limiter.acquire(300)
    assert clock.sleeps == []

    limiter.acquire(500)  # 900 used; must wait for the 600-token request to expire
    assert sum(clock.sleeps) == pytest.approx(50.0)


def test_record_usage_frees_overestimated_tokens(clock):
    limiter = RateLimiter("test", tpm=1000)
    event = limiter.acquire(900)          # estimate (prompt/4 + max_tokens)
    limiter.record_usage(event, 200)      # the API billed far less
    limiter.acquire(700)
    assert clock.sleeps == []


def test_cancel_releases_reservation(clock):
    limiter = RateLimiter("test", tpm=1000)
    event = limiter.acquire(900)
    limiter.cancel(event)                 # the request was rejected
    limiter.acquire(900)
    assert clock.sleeps == []


def test_block_for_pauses_everyone(clock):
    limiter = RateLimiter("test")         # no rpm/tpm limits at all
    limiter.block_for(12)
    limiter.acquire(1)
    assert sum(clock.sleeps) == pytest.approx(12.0)


def test_oversized_request_waits_for_empty_window_then_goes(clock):
    limiter = RateLimiter("test", tpm=100)
    limiter.acquire(50)
    limiter.acquire(500)                  # bigger than the whole budget
    assert sum(clock.sleeps) == pytest.approx(60.0)


# ── 429 parsing ───────────────────────────────────────────────────────────────

def _rate_limit_error(message, headers=None):
    request = httpx.Request("POST", "https://api.example/v1/chat/completions")
    response = httpx.Response(429, headers=headers or {}, request=request)
    return openai.RateLimitError(message, response=response, body=None)


@pytest.mark.parametrize("message, expected", [
    ("Rate limit reached. Please try again in 7.66s.", 8.66),
    ("Please try again in 1m26.4s", 87.4),
    ("retry in 340ms", 1.34),
    ("Please try again in 2h3m", 7381.0),
])
def test_retry_after_from_message(message, expected):
    assert _retry_after_seconds(_rate_limit_error(message)) == pytest.approx(expected)


def test_retry_after_header_wins():
    exc = _rate_limit_error("try again in 99s", headers={"retry-after": "4"})
    assert _retry_after_seconds(exc) == pytest.approx(5.0)


def test_no_hint_returns_none():
    assert _retry_after_seconds(_rate_limit_error("slow down")) is None


@pytest.mark.parametrize("message, daily", [
    ("Limit 200000, Used 199000 on tokens per day (TPD)", True),
    ("requests per day exceeded", True),
    ("Limit 6000 on tokens per minute (TPM)", False),
])
def test_daily_quota_detection(message, daily):
    assert _is_daily_quota(_rate_limit_error(message)) is daily


# ── chat() retry loop ─────────────────────────────────────────────────────────

class _FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def create(self, **request):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        usage = type("Usage", (), {"total_tokens": 42})()
        message = type("Msg", (), {"content": outcome})()
        choice = type("Choice", (), {"message": message})()
        return type("Resp", (), {"usage": usage, "choices": [choice]})()


@pytest.fixture
def fake_provider(monkeypatch):
    """Point chat() at a fake OpenAI client and a limiter with a fake clock."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    limiter = RateLimiter("openai")
    blocked = []
    monkeypatch.setattr(limiter, "block_for", blocked.append)
    monkeypatch.setattr(llm_client, "_get_limiter", lambda provider: limiter)
    monkeypatch.setattr(llm_client, "time", SimpleNamespace(sleep=lambda s: None))

    def install(*outcomes):
        completions = _FakeCompletions(outcomes)
        fake_client = type("Client", (), {})()
        fake_client.chat = type("Chat", (), {"completions": completions})()
        monkeypatch.setattr(llm_client, "_get_client", lambda provider: fake_client)
        return completions, blocked
    return install


def test_chat_retries_after_429_using_server_hint(fake_provider):
    completions, blocked = fake_provider(_rate_limit_error("try again in 2s"), "hello")
    assert llm_client.chat("sys", "user", provider="openai") == "hello"
    assert completions.calls == 2
    assert blocked == [pytest.approx(3.0)]


def test_chat_stops_on_daily_quota(fake_provider):
    fake_provider(_rate_limit_error("tokens per day (TPD) exceeded, try again in 3h"))
    with pytest.raises(RateLimitExhausted) as info:
        llm_client.chat("sys", "user", provider="openai")
    assert info.value.provider == "openai"


def test_chat_stops_when_wait_is_too_long(fake_provider, monkeypatch):
    monkeypatch.setenv("LLM_MAX_WAIT_SECONDS", "30")
    fake_provider(_rate_limit_error("try again in 5m"))
    with pytest.raises(RateLimitExhausted):
        llm_client.chat("sys", "user", provider="openai")


def test_chat_retries_transient_errors(fake_provider):
    request = httpx.Request("POST", "https://api.example")
    completions, _ = fake_provider(openai.APIConnectionError(request=request), "ok")
    assert llm_client.chat("sys", "user", provider="openai") == "ok"
    assert completions.calls == 2


def test_resolve_provider_without_keys_explains_setup():
    with pytest.raises(EnvironmentError, match="No LLM configured"):
        llm_client.resolve_provider("auto")
