"""Errors raised by aibridge-core. Same class and codes as aibridge_exec.providers.errors; messages never contain the key."""
from __future__ import annotations

from typing import Literal

ErrorCode = Literal[
    "auth",            # key missing, invalid or not allowed for this model
    "rate_limited",    # provider throttled the request or quota exhausted
    "bad_request",     # provider rejected the request (unknown model, bad parameters)
    "not_found",       # model or endpoint does not exist
    "upstream",        # provider-side failure (5xx, overloaded)
    "timeout",
    "network",
    "unknown_provider",
    "config",
]

HTTP_STATUS_FOR_CODE: dict[str, int] = {
    "auth": 401,
    "rate_limited": 429,
    "bad_request": 400,
    "not_found": 404,
    "upstream": 502,
    "timeout": 504,
    "network": 502,
    "unknown_provider": 404,
    "config": 400,
}


class ProviderError(Exception):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        provider: str | None = None,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.provider = provider
        self.status = status

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "provider": self.provider,
            "upstream_status": self.status,
        }

    def __repr__(self) -> str:
        return f"ProviderError(code={self.code!r}, provider={self.provider!r}, message={self.message!r})"


def code_for_status(status: int) -> ErrorCode:
    if status in (401, 403):
        return "auth"
    if status == 429:
        return "rate_limited"
    if status == 404:
        return "not_found"
    if 400 <= status < 500:
        return "bad_request"
    return "upstream"


def redact(text: str, secret: str | None) -> str:
    if secret and len(secret) >= 6:
        text = text.replace(secret, "***")
    return text
