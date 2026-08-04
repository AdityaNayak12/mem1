class SemanticReviewError(Exception):
    """Base exception for all semantic validation and review failures."""
    pass


class ReviewerTimeout(SemanticReviewError):
    """Raised when the reviewer LLM request times out."""
    pass


class InvalidReviewerResponse(SemanticReviewError):
    """Raised when the reviewer LLM returns a malformed response or fails validation."""
    pass
