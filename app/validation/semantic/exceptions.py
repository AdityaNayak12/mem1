class SemanticValidationError(Exception):
    """Base exception for semantic validation failures."""
    pass


class ReviewerError(SemanticValidationError):
    """Raised when the LLM reviewer experiences an API or provider failure."""
    pass


class InvalidReviewResponseError(SemanticValidationError):
    """Raised when the LLM reviewer's output is malformed or invalid."""
    pass
