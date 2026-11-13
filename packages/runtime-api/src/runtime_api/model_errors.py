"""Why a model call failed, as the code a failure reply reads, and its detail for logs.

The run-completion webhook only ever sees the exception's class name (the platform
scrubs the message of any type outside its allowlist), so the classification that
tells "provider overloaded" from every other ``APIError`` is made where the call
failed, while the exception is intact, and left on the thread for the reply to read.
"""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

_MAX_BODY_CHARS: Final = 2000

# Thread-metadata key holding the classification of the run's last model error,
# written while the exception is intact and read back by the completion webhook.
LAST_MODEL_ERROR_KEY: Final = "last_model_error"

# The ``response_metadata`` key marking an answer the runtime wrote for a model call it
# could not make (``outage`` or ``model_access``). It is final: the loop posts it as it
# is instead of asking the model again for a reply it cannot give.
MODEL_FAILURE_REPLY_KEY: Final = "open_swe_model_failure"

# Exception class names, the only failure signal the run-completion webhook gets:
# the platform serializes the class name but scrubs the message of any exception
# type outside its allowlist.
_ERROR_TYPE_CODES: Final = {
    "AuthenticationError": "model_unavailable",
    "PermissionDeniedError": "model_unavailable",
    "OpenAIAuthenticationError": "model_unavailable",
    "OpenAIPermissionDeniedError": "model_unavailable",
    "RateLimitError": "provider_rate_limited",
    "OverloadedError": "provider_overloaded",
    "APIConnectionError": "provider_unavailable",
    "APIError": "provider_unavailable",
    "APIStatusError": "provider_unavailable",
    "InternalServerError": "provider_unavailable",
    "APITimeoutError": "provider_timeout",
    "ModelCallTimeoutError": "provider_timeout",
    "GraphRecursionError": "step_limit",
    "RecursionError": "step_limit",
    "SandboxConnectionError": "sandbox_unreachable",
    "SandboxUnreachableError": "sandbox_unreachable",
    "BackendUnavailableError": "backend_interrupted",
    "BackendTimeoutError": "backend_interrupted",
    # Not raised: the stalled-run sweep names it when it fails a run that went silent.
    "RunStalledError": "run_stalled",
}


@dataclass(frozen=True)
class ModelAccessFailure:
    reason: Literal["secret", "policy", "key", "permission"]
    secret: str | None = None
    provider_scoped: bool = False

    @classmethod
    def from_exception(cls, exc: BaseException) -> ModelAccessFailure | None:
        status = getattr(exc, "status_code", None)
        body = getattr(exc, "body", None)
        if (
            not isinstance(status, int)
            and type(exc).__name__ not in _ERROR_TYPE_CODES
            and not isinstance(body, Mapping)
        ):
            return None
        detail = f"{body} {exc}"
        lowered = detail.lower()
        missing_secret = "secret" in lowered and any(
            word in lowered
            for word in ("missing", "not found", "does not exist", "not configured", "not set")
        )
        policy = any(
            word in lowered
            for word in (
                "policy",
                "spend limit",
                "spend_limit",
                "spending limit",
                "budget exceeded",
            )
        )
        if missing_secret:
            match = re.search(
                r"(?i:secret(?:_name)?(?: with name)?)['\"]?\s*[:=]?\s*['\"`]?([A-Z][A-Z0-9_]{0,127})(?=['\"`\s,}]|$)",
                detail,
            )
            secret = match.group(1) if match else None
            return cls(
                "secret",
                secret,
                bool(secret and secret.startswith(("ANTHROPIC_", "OPENAI_", "FIREWORKS_"))),
            )
        if status == 402 or policy:
            return cls("policy")
        if status == 401 or type(exc).__name__ in {
            "AuthenticationError",
            "OpenAIAuthenticationError",
        }:
            return cls("key", provider_scoped="langsmith" not in lowered)
        if status in {403, 424} or type(exc).__name__ in {
            "PermissionDeniedError",
            "OpenAIPermissionDeniedError",
        }:
            return cls(
                "permission",
                provider_scoped=status == 403
                and "langsmith" not in lowered
                and any(
                    word in lowered
                    for word in (
                        "provider key",
                        "provider permission",
                        "api key",
                        "permission_error",
                    )
                ),
            )
        return None

    def message(self) -> str:
        if self.reason == "secret":
            name = f" `{self.secret}`" if self.secret else " required by the selected model"
            return (
                f"The workspace secret{name} is missing. "
                "A workspace admin must add it in LangSmith workspace secrets. "
                "Choose another model or contact your workspace admin."
            )
        if self.reason == "policy":
            return (
                "A LangSmith gateway policy or spend limit blocked this model call. "
                "Choose another model or ask your workspace admin to review the gateway policy and spend limits."
            )
        if self.reason == "key":
            return (
                "The provider key configured for the workspace was rejected. "
                "Ask your workspace admin to update the workspace credentials, or choose another model."
            )
        return (
            "The workspace does not have permission to call the selected model. "
            "Choose another model or ask your workspace admin to check the workspace credentials and model access."
        )


def is_model_failure_reply(message: object) -> bool:
    """Whether ``message`` is the runtime's answer for a model call it could not make."""
    metadata = getattr(message, "response_metadata", None)
    return isinstance(metadata, Mapping) and bool(metadata.get(MODEL_FAILURE_REPLY_KEY))


def code_for_error_type(error_type: str | None) -> str | None:
    """Failure code for a bare exception class name."""
    return _ERROR_TYPE_CODES.get(error_type) if error_type else None


def classify_exception(exc: BaseException) -> str | None:
    """Failure code for a live exception, using status and provider error body."""
    if ModelAccessFailure.from_exception(exc) is not None:
        return "model_unavailable"
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    provider_code = ""
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict):
            provider_code = str(inner.get("type") or inner.get("code") or "")
    haystack = f"{provider_code} {exc}".lower()

    if status == 429 or "rate_limit" in haystack:
        return "provider_rate_limited"
    if status == 529 or "overloaded" in haystack:
        return "provider_overloaded"
    if "context_length_exceeded" in haystack or "prompt is too long" in haystack:
        return "context_too_long"
    if "model_not_available" in haystack or "model_not_found" in haystack:
        return "model_unavailable"
    if isinstance(status, int) and status >= 500:
        return "provider_unavailable"
    return code_for_error_type(type(exc).__name__)


def exception_fields(exc: BaseException) -> dict[str, Any]:
    """Class, message, and any HTTP status / request id / response body on ``exc``.

    Provider SDKs put the useful part of a failure (``overloaded_error``, a
    rate-limit window, a rejected model id) in the response body, which
    ``str(exc)`` frequently omits. For logs only: the body is bounded, never shown.
    """
    try:
        fields: dict[str, Any] = {"error_type": type(exc).__name__, "error_message": str(exc)}
        status = getattr(exc, "status_code", None)
        if status is not None:
            fields["status"] = status
        request_id = getattr(exc, "request_id", None)
        if request_id:
            fields["request_id"] = request_id
        body = getattr(exc, "body", None)
        if body is not None:
            fields["body"] = _bounded(body)
        cause = exc.__cause__ or exc.__context__
        if cause is not None:
            fields["cause"] = f"{type(cause).__name__}: {cause}"
        return fields
    except Exception:  # noqa: BLE001
        return {"error_type": type(exc).__name__, "error_message": repr(exc)}


def _bounded(body: Any) -> Any:
    try:
        rendered = json.dumps(body, default=str)
    except TypeError, ValueError:
        rendered = str(body)
    if len(rendered) <= _MAX_BODY_CHARS:
        return body
    return rendered[:_MAX_BODY_CHARS]
