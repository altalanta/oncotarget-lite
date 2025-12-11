import asyncio

import pytest

from oncotarget_lite.retry import retry
from oncotarget_lite.rate_limit import RateLimiter


def test_retry_eventually_succeeds():
    call_count = {"n": 0}

    @retry(max_attempts=3, base_delay=0.01, max_delay=0.05)
    def sometimes_fails():
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ConnectionError("temporary")
        return "ok"

    result = sometimes_fails()
    assert result == "ok"
    assert call_count["n"] == 3  # exhausted two retries then succeeded


def test_retry_exhausts():
    @retry(max_attempts=2, base_delay=0.01, max_delay=0.05, exceptions=(ValueError,))
    def always_fails():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        always_fails()


@pytest.mark.asyncio
async def test_rate_limiter_enforces_burst():
    limiter = RateLimiter(requests_per_minute=2, burst_size=2, cleanup_interval=10)
    key = "client-1"

    allowed1, _ = await limiter.is_allowed(key)
    allowed2, _ = await limiter.is_allowed(key)
    # Third request should be limited because burst tokens are consumed
    allowed3, retry_after = await limiter.is_allowed(key)

    assert allowed1 is True
    assert allowed2 is True
    assert allowed3 is False
    assert retry_after > 0

