"""Provider-neutral chat types, identical to aibridge_exec.providers.types (same fields, validation and to_dict)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping

Role = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length", "content_filter", "tool_calls", "error", "other"]

ROLES = ("system", "user", "assistant")


@dataclass(frozen=True)
class ChatMessage:
    """One turn. ``cache=True`` marks the conversation up to and including this message as a
    stable prefix worth caching; adapters with opt-in prompt caching (Anthropic) add a cache
    breakpoint there, the others ignore it."""

    role: Role
    content: str
    cache: bool = False

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")

    @classmethod
    def coerce(cls, value: "ChatMessage | Mapping[str, Any]") -> "ChatMessage":
        if isinstance(value, ChatMessage):
            return value
        if not isinstance(value, Mapping) and hasattr(value, "role") and hasattr(value, "content"):
            # e.g. aibridge_exec.providers.ChatMessage, so both libraries can be mixed while migrating
            return cls(role=value.role, content=value.content, cache=bool(getattr(value, "cache", False)))
        return cls(role=value["role"], content=value["content"], cache=bool(value.get("cache", False)))


REASONING_EFFORTS = ("lowest", "none", "minimal", "low", "medium", "high", "xhigh")


@dataclass(frozen=True)
class ResponseFormat:
    """Ask for a JSON reply. Each adapter uses its provider's native mechanism (a forced tool,
    a JSON-schema response format, JSON mode or a schema-constrained decoder) and puts the JSON
    text in ``ChatResponse.text``. Providers without one reply as instructed by the prompt.

    ``schema`` is a JSON Schema for the reply (``None``: any JSON object).
    """

    schema: Mapping[str, Any] | None = None
    name: str = "Give_Information"
    description: str = "Generate output in given key and valid json format"

    def __post_init__(self) -> None:
        if self.schema is not None and not isinstance(self.schema, Mapping):
            raise TypeError("response_format schema must be a mapping (JSON Schema)")
        if not self.name or not all(c.isalnum() or c in "_-" for c in self.name):
            raise ValueError("response_format name must be letters, digits, '_' or '-'")

    @classmethod
    def coerce(cls, value: "ResponseFormat | Mapping[str, Any] | None") -> "ResponseFormat | None":
        if value is None or isinstance(value, ResponseFormat):
            return value
        if not isinstance(value, Mapping) and hasattr(value, "schema") and hasattr(value, "name"):
            # aibridge_exec.providers.ResponseFormat
            return cls(schema=value.schema, name=value.name, description=value.description)
        return cls(**dict(value))

    def json_schema(self) -> dict[str, Any]:
        """The schema to send; an open object when none was given."""
        return dict(self.schema) if self.schema else {"type": "object"}


@dataclass(frozen=True)
class ChatRequest:
    """One chat completion call. ``system`` and any ``system`` messages are merged.

    Optional, provider-neutral hints (adapters that can't use one ignore it):
    ``response_format`` asks for JSON (see ``ResponseFormat``); ``reasoning_effort`` sets how hard reasoning models think (``"lowest"`` picks the cheapest
    setting the model supports); ``thinking`` turns a hybrid model's thinking mode on or off.
    """

    model: str
    messages: tuple[ChatMessage, ...]
    system: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: ResponseFormat | None = None
    reasoning_effort: str | None = None
    thinking: bool | None = None

    def __post_init__(self) -> None:
        if not self.model or not isinstance(self.model, str):
            raise ValueError("model is required")
        if not self.messages:
            raise ValueError("at least one message is required")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.reasoning_effort is not None and self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS}")

    @classmethod
    def build(
        cls,
        model: str,
        messages: Iterable["ChatMessage | Mapping[str, Any]"],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: "ResponseFormat | Mapping[str, Any] | None" = None,
        reasoning_effort: str | None = None,
        thinking: bool | None = None,
    ) -> "ChatRequest":
        return cls(
            model=model,
            messages=tuple(ChatMessage.coerce(m) for m in messages),
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=ResponseFormat.coerce(response_format),
            reasoning_effort=reasoning_effort,
            thinking=thinking,
        )

    def system_parts(self) -> tuple[list[ChatMessage], list[ChatMessage]]:
        """System messages (``system`` first) kept separate, for adapters that send blocks, and the turns."""
        parts = [ChatMessage("system", self.system)] if self.system else []
        parts += [m for m in self.messages if m.role == "system" and m.content]
        return parts, [m for m in self.messages if m.role != "system"]

    def split_system(self) -> tuple[str | None, list[ChatMessage]]:
        """System text (from ``system`` plus system messages) and the remaining turns."""
        parts = [self.system] if self.system else []
        turns: list[ChatMessage] = []
        for message in self.messages:
            if message.role == "system":
                parts.append(message.content)
            else:
                turns.append(message)
        return ("\n\n".join(p for p in parts if p) or None), turns


def merge_consecutive(turns: list[ChatMessage]) -> list[ChatMessage]:
    """Join back-to-back turns with the same role (providers that require alternation)."""
    merged: list[ChatMessage] = []
    for turn in turns:
        if merged and merged[-1].role == turn.role:
            merged[-1] = ChatMessage(turn.role, merged[-1].content + "\n\n" + turn.content,
                                     cache=merged[-1].cache or turn.cache)
        else:
            merged.append(turn)
    return merged


@dataclass
class Usage:
    """Token counts. ``input_tokens`` is the whole prompt, cached parts included; when the provider
    reports prompt caching, ``cache_read_tokens`` / ``cache_write_tokens`` say how much of it was
    read from or written to the cache (they appear in ``to_dict`` only when known)."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        out = {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}
        if self.cache_read_tokens is not None:
            out["cache_read_tokens"] = self.cache_read_tokens
        if self.cache_write_tokens is not None:
            out["cache_write_tokens"] = self.cache_write_tokens
        return out


@dataclass
class ChatResponse:
    text: str
    model: str
    finish_reason: FinishReason | None = None
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict(),
        }


@dataclass
class StreamEvent:
    """``delta`` events carry text; one final ``done`` event carries usage."""

    type: Literal["delta", "done"]
    text: str = ""
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.type == "delta":
            return {"type": "delta", "text": self.text}
        return {
            "type": "done",
            "finish_reason": self.finish_reason,
            "model": self.model,
            "usage": (self.usage or Usage()).to_dict(),
        }


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str | None = None
    context_length: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name or self.id, "context_length": self.context_length}


@dataclass(frozen=True)
class ProviderInfo:
    id: str
    label: str
    key_url: str
    default_base_url: str
    key_hint: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "label": self.label, "key_url": self.key_url, "key_hint": self.key_hint}
