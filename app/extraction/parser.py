import json
from typing import Any
from pydantic import ValidationError

from app.schemas.memory_ir import MemoryIR
from app.extraction.exceptions import InvalidResponseError


def parse_response(response: str | dict[str, Any] | Any) -> MemoryIR:
    """Deserializes LLM output into a validated MemoryIR object.

    Args:
        response: Raw JSON string, a dictionary, or a Pydantic-compatible object.

    Returns:
        MemoryIR: A validated MemoryIR model instance.

    Raises:
        InvalidResponseError: If parsing fails or fails schema validation.
    """
    if not response:
        raise InvalidResponseError("Empty response received from provider")

    # If the response is already a dict or a BaseModel
    if isinstance(response, dict):
        try:
            return MemoryIR.model_validate(response)
        except ValidationError as e:
            raise InvalidResponseError(f"Pydantic validation failed: {e}") from e

    # If it is a string (JSON representation)
    if isinstance(response, str):
        cleaned_response = response.strip()
        
        # Strip potential markdown block syntax (e.g. ```json ... ```)
        if cleaned_response.startswith("```"):
            lines = cleaned_response.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned_response = "\n".join(lines).strip()

        try:
            return MemoryIR.model_validate_json(cleaned_response)
        except (json.JSONDecodeError, ValueError) as e:
            raise InvalidResponseError(f"Failed to parse JSON string: {e}") from e
        except ValidationError as e:
            raise InvalidResponseError(f"Pydantic validation failed: {e}") from e

    # If it's a Pydantic model (e.g. if LiteLLM structured outputs returns parsed object)
    if hasattr(response, "model_validate"):
        try:
            # Re-validate to ensure type safety
            return MemoryIR.model_validate(response)
        except ValidationError as e:
            raise InvalidResponseError(f"Pydantic validation failed: {e}") from e

    # General fallback validation
    try:
        return MemoryIR.model_validate(response)
    except ValidationError as e:
        raise InvalidResponseError(f"Pydantic validation failed: {e}") from e
