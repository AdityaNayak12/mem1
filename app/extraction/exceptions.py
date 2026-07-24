class ExtractionError(Exception):
    pass


class ProviderError(ExtractionError):
    """Raised when the LLM provider fails (e.g., API errors, rate limits, timeouts)."""
    pass


class InvalidResponseError(ExtractionError):
    """Raised when the LLM response is malformed, invalid JSON, or fails schema validation."""
    pass
