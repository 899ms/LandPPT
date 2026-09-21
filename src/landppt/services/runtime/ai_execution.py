"""
Explicit execution context objects for provider-bound workflows.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import uuid
from typing import Any, Dict, Optional


_current_ai_conversation_id: ContextVar[Optional[str]] = ContextVar(
    "landppt_ai_conversation_id", default=None
)


def new_ai_conversation_id(prefix: str = "landppt") -> str:
    """Create an identity at a logical workflow boundary, never per provider request."""
    return f"{prefix}-{uuid.uuid4().hex}"


def get_current_ai_conversation_id() -> Optional[str]:
    return _current_ai_conversation_id.get()


def is_provider_protocol_error(error: Exception) -> bool:
    """Return whether an error represents a provider/protocol failure."""
    seen = set()
    current = error
    markers = (
        "MissingSessionID", "x-opencode-session", "Error from provider",
        "APIStatusError", "BadRequestError", "AuthenticationError",
        "PermissionDeniedError", "RateLimitError", "InternalServerError",
        "HTTP 400", "HTTP 401", "HTTP 403", "HTTP 429", "HTTP 500",
    )
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        class_name = type(current).__name__
        module_name = type(current).__module__
        message = str(current)
        if module_name.startswith("openai") or any(
            marker in class_name or marker in message for marker in markers
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


@contextmanager
def ai_conversation_context(conversation_id: Optional[str]):
    """Propagate an owner-provided conversation identity across async call layers."""
    token = _current_ai_conversation_id.set(conversation_id)
    try:
        yield conversation_id
    finally:
        _current_ai_conversation_id.reset(token)


def scoped_ai_conversation(prefix: str):
    """Scope a top-level logical operation without storing session state on services."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            conversation_id = (
                kwargs.get("conversation_id")
                or next(
                    (
                        getattr(argument, "conversation_id", None)
                        for argument in args
                        if getattr(argument, "conversation_id", None)
                    ),
                    None,
                )
                or get_current_ai_conversation_id()
                or new_ai_conversation_id(prefix)
            )
            with ai_conversation_context(conversation_id):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    use_responses_api: bool = False
    enable_reasoning: bool = False
    reasoning_effort: str = "medium"
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "ProviderConfig":
        return cls(
            provider=str(data.get("llm_provider") or data.get("provider") or "openai"),
            model=data.get("llm_model") or data.get("model"),
            api_key=data.get("api_key"),
            base_url=data.get("base_url"),
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
            use_responses_api=bool(data.get("use_responses_api")),
            enable_reasoning=bool(data.get("enable_reasoning")),
            reasoning_effort=str(data.get("reasoning_effort") or "medium"),
            extras={
                key: value
                for key, value in data.items()
                if key
                not in {
                    "llm_provider",
                    "provider",
                    "llm_model",
                    "model",
                    "api_key",
                    "base_url",
                    "temperature",
                    "max_tokens",
                    "use_responses_api",
                    "enable_reasoning",
                    "reasoning_effort",
                }
            },
        )


@dataclass(frozen=True)
class ExecutionContext:
    role: str
    provider: ProviderConfig
    user_id: Optional[int] = None
    source: str = "resolved"
    conversation_id: Optional[str] = None

    @classmethod
    def from_mapping(
        cls,
        role: str,
        data: Dict[str, Any],
        *,
        user_id: Optional[int] = None,
        source: str = "resolved",
        conversation_id: Optional[str] = None,
    ) -> "ExecutionContext":
        return cls(
            role=role,
            provider=ProviderConfig.from_mapping(data),
            user_id=user_id,
            source=source,
            conversation_id=conversation_id,
        )

    def to_processing_config_kwargs(self) -> Dict[str, Any]:
        result = {
            "llm_model": self.provider.model,
            "llm_provider": self.provider.provider,
            "temperature": self.provider.temperature,
            "max_tokens": self.provider.max_tokens,
            "api_key": self.provider.api_key,
            "base_url": self.provider.base_url,
            "use_responses_api": self.provider.use_responses_api,
            "enable_reasoning": self.provider.enable_reasoning,
            "reasoning_effort": self.provider.reasoning_effort,
        }
        if self.conversation_id:
            result["conversation_id"] = self.conversation_id
        return result

