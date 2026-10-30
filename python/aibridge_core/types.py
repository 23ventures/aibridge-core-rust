"""Provider-neutral chat types, identical to aibridge_exec.providers.types (same fields, validation and to_dict)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal, Mapping

Role = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length", "content_filter", "tool_calls", "error", "other"]

ROLES = ("system", "user", "assistant")


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str

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
            return cls(role=value.role, content=value.content)
        return cls(role=value["role"], content=value["content"])


@dataclass(frozen=True)
class ChatRequest:
    """One chat completion call. ``system`` and any ``system`` messages are merged."""

    model: str
    messages: tuple[ChatMessage, ...]
    system: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.model or not isinstance(self.model, str):
            raise ValueError("model is required")
        if not self.messages:
            raise ValueError("at least one message is required")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")

    @classmethod
    def build(
        cls,
        model: str,
        messages: Iterable["ChatMessage | Mapping[str, Any]"],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> "ChatRequest":
        return cls(
            model=model,
            messages=tuple(ChatMessage.coerce(m) for m in messages),
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

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
            merged[-1] = ChatMessage(turn.role, merged[-1].content + "\n\n" + turn.content)
        else:
            merged.append(turn)
    return merged


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)


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
