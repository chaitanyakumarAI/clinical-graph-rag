"""
llm/rate_limiter.py
Client-side throttle that keeps us under a provider's per-minute limits.

Free-tier LLM APIs (Groq, Gemini) enforce two limits over a rolling minute:
  RPM — requests per minute
  TPM — tokens per minute (prompt + completion)

Instead of firing requests and waiting for HTTP 429 errors, every call first
asks this limiter for permission. If sending now would exceed either limit,
the limiter sleeps just long enough for old requests to fall out of the window.

The limiter is thread-safe, so concurrent FastAPI requests share one budget.
"""

from __future__ import annotations
import threading
import time
from collections import deque
from typing import Deque, List

WINDOW_SECONDS = 60.0


class RateLimiter:
    def __init__(self, name: str, rpm: int = 0, tpm: int = 0):
        """
        Args:
            name: Provider name, used only in log messages.
            rpm:  Max requests per minute (0 = unlimited).
            tpm:  Max tokens per minute   (0 = unlimited).
        """
        self.name = name
        self.rpm  = rpm
        self.tpm  = tpm
        # Each entry is a mutable [timestamp, tokens] pair so the token count
        # can be corrected once the real usage is known.
        self._events: Deque[List[float]] = deque()
        self._blocked_until = 0.0
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def acquire(self, tokens: int) -> List[float]:
        """
        Block until a request of `tokens` tokens fits within the limits,
        then reserve it. Returns a handle for record_usage() / cancel().
        """
        while True:
            with self._lock:
                now = time.monotonic()
                self._forget_old_events(now)
                wait = self._seconds_until_allowed(now, tokens)
                if wait <= 0:
                    event = [now, float(tokens)]
                    self._events.append(event)
                    return event

            if wait > 2:
                print(f"  [rate-limit] {self.name}: pausing {wait:.0f}s to stay under "
                      f"{self.rpm or '∞'} req/min, {self.tpm or '∞'} tokens/min")
            time.sleep(wait)

    def record_usage(self, event: List[float], actual_tokens: int) -> None:
        """Replace the estimated token count with what the API actually billed."""
        with self._lock:
            event[1] = float(actual_tokens)

    def cancel(self, event: List[float]) -> None:
        """A rejected request consumed no tokens — free its reservation."""
        with self._lock:
            event[1] = 0.0

    def block_for(self, seconds: float) -> None:
        """Stop all callers for `seconds` (used after the server returns 429)."""
        with self._lock:
            self._blocked_until = max(self._blocked_until, time.monotonic() + seconds)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _forget_old_events(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= WINDOW_SECONDS:
            self._events.popleft()

    def _seconds_until_allowed(self, now: float, tokens: int) -> float:
        waits = [self._blocked_until - now]

        # Requests/min: the oldest request that must expire before one more fits.
        if self.rpm and len(self._events) >= self.rpm:
            oldest_blocking = self._events[len(self._events) - self.rpm]
            waits.append(oldest_blocking[0] + WINDOW_SECONDS - now)

        # Tokens/min: expire old requests (oldest first) until the new one fits.
        if self.tpm:
            used = sum(tokens_ for _, tokens_ in self._events)
            if used + tokens > self.tpm:
                wait_tokens = 0.0
                for timestamp, tokens_ in self._events:
                    used -= tokens_
                    wait_tokens = timestamp + WINDOW_SECONDS - now
                    if used + tokens <= self.tpm:
                        break
                # A single request bigger than the whole budget is sent once
                # the window is empty; the server decides whether to accept it.
                waits.append(wait_tokens)

        return max(waits)
