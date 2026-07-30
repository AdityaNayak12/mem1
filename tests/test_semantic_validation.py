import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.memory_ir import MemoryIR
from app.validation import IssueLevel
from app.validation.semantic import (
    SemanticValidator,
    MemoryReviewer,
    LiteLLMMemoryReviewer,
    SemanticReview,
    ReviewFinding,
    ReviewerError,
    InvalidReviewResponseError,
)


class MockReviewer(MemoryReviewer):
    """Mock implementation of the MemoryReviewer protocol."""

    def __init__(self):
        self.review_mock = AsyncMock()

    async def review(self, conversation: str, memory: MemoryIR) -> SemanticReview:
        return await self.review_mock(conversation, memory)


@pytest.fixture
def mock_reviewer():
    return MockReviewer()


@pytest.fixture
def validator(mock_reviewer):
    return SemanticValidator(reviewer=mock_reviewer)


@pytest.fixture
def empty_memory():
    return MemoryIR(entities=[], relationships=[], events=[])


@pytest.mark.asyncio
async def test_valid_memory(validator, mock_reviewer, empty_memory):
    # Setup mock to return a review with no findings
    mock_reviewer.review_mock.return_value = SemanticReview(findings=[])

    result = await validator.validate("User: Hello", empty_memory)

    assert result.valid is True
    assert len(result.issues) == 0
    mock_reviewer.review_mock.assert_called_once_with("User: Hello", empty_memory)


@pytest.mark.asyncio
async def test_unsupported_entity(validator, mock_reviewer, empty_memory):
    mock_reviewer.review_mock.return_value = SemanticReview(
        findings=[
            ReviewFinding(
                finding_type="unsupported_entity",
                severity="ERROR",
                location="entities[0]",
                explanation="Entity not mentioned.",
                recommendation="Remove entity."
            )
        ]
    )

    result = await validator.validate("User: Hello", empty_memory)

    assert result.valid is False
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.level == IssueLevel.ERROR
    assert issue.code == "SEMANTIC_UNSUPPORTED_ENTITY"
    assert issue.location == "entities[0]"
    assert "Entity not mentioned." in issue.message
    assert "Remove entity." in issue.message


@pytest.mark.asyncio
async def test_unsupported_relationship(validator, mock_reviewer, empty_memory):
    mock_reviewer.review_mock.return_value = SemanticReview(
        findings=[
            ReviewFinding(
                finding_type="unsupported_relationship",
                severity="ERROR",
                location="relationships[0]",
                explanation="Relationship not direct.",
                recommendation="Fix predicate."
            )
        ]
    )

    result = await validator.validate("User: Hello", empty_memory)

    assert result.valid is False
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.level == IssueLevel.ERROR
    assert issue.code == "SEMANTIC_UNSUPPORTED_RELATIONSHIP"
    assert issue.location == "relationships[0]"


@pytest.mark.asyncio
async def test_unsupported_event(validator, mock_reviewer, empty_memory):
    mock_reviewer.review_mock.return_value = SemanticReview(
        findings=[
            ReviewFinding(
                finding_type="unsupported_event",
                severity="ERROR",
                location="events[0]",
                explanation="No such event.",
                recommendation="Remove event."
            )
        ]
    )

    result = await validator.validate("User: Hello", empty_memory)

    assert result.valid is False
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.level == IssueLevel.ERROR
    assert issue.code == "SEMANTIC_UNSUPPORTED_EVENT"
    assert issue.location == "events[0]"


@pytest.mark.asyncio
async def test_hallucination(validator, mock_reviewer, empty_memory):
    mock_reviewer.review_mock.return_value = SemanticReview(
        findings=[
            ReviewFinding(
                finding_type="hallucination",
                severity="ERROR",
                location="entities[1]",
                explanation="Invented info.",
                recommendation="Remove hallucinated properties."
            )
        ]
    )

    result = await validator.validate("User: Hello", empty_memory)

    assert result.valid is False
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.level == IssueLevel.ERROR
    assert issue.code == "SEMANTIC_HALLUCINATION"
    assert issue.location == "entities[1]"


@pytest.mark.asyncio
async def test_missing_memory(validator, mock_reviewer, empty_memory):
    mock_reviewer.review_mock.return_value = SemanticReview(
        findings=[
            ReviewFinding(
                finding_type="missing_memory",
                severity="WARNING",
                location="general",
                explanation="Omitted Google employment details.",
                recommendation="Extract Google company entity."
            )
        ]
    )

    result = await validator.validate("User: Hello", empty_memory)

    # Warnings alone must not invalidate the overall result
    assert result.valid is True
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.level == IssueLevel.WARNING
    assert issue.code == "SEMANTIC_MISSING_MEMORY"
    assert issue.location == "general"


@pytest.mark.asyncio
async def test_malformed_review_exception(validator, mock_reviewer, empty_memory):
    mock_reviewer.review_mock.side_effect = InvalidReviewResponseError("Malformed JSON structure")

    result = await validator.validate("User: Hello", empty_memory)

    assert result.valid is False
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.level == IssueLevel.ERROR
    assert issue.code == "SEMANTIC_REVIEW_FAILURE"
    assert "Malformed JSON structure" in issue.message


@pytest.mark.asyncio
async def test_timeout_exception(validator, mock_reviewer, empty_memory):
    mock_reviewer.review_mock.side_effect = ReviewerError("LLM call timed out")

    result = await validator.validate("User: Hello", empty_memory)

    assert result.valid is False
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.level == IssueLevel.ERROR
    assert issue.code == "SEMANTIC_REVIEW_FAILURE"
    assert "LLM call timed out" in issue.message


@pytest.mark.asyncio
async def test_provider_failure_exception(validator, mock_reviewer, empty_memory):
    mock_reviewer.review_mock.side_effect = ReviewerError("API connection failure")

    result = await validator.validate("User: Hello", empty_memory)

    assert result.valid is False
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.level == IssueLevel.ERROR
    assert issue.code == "SEMANTIC_REVIEW_FAILURE"
    assert "API connection failure" in issue.message


# --- LiteLLMMemoryReviewer Specific Tests ---

def test_litellm_reviewer_initialization():
    with patch.dict("os.environ", {"MEM1_VALIDATOR_MODEL": "openrouter/google/gemini-2.5-flash-lite"}):
        reviewer = LiteLLMMemoryReviewer()
        assert reviewer.model == "openrouter/google/gemini-2.5-flash-lite"
        assert reviewer.provider == "openrouter"


@pytest.mark.asyncio
async def test_litellm_reviewer_missing_api_key(empty_memory):
    reviewer = LiteLLMMemoryReviewer(model="openrouter/google/gemini-2.5-flash-lite")
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ReviewerError) as exc_info:
            await reviewer.review("User: Hello", empty_memory)
        assert "Missing OPENROUTER_API_KEY" in str(exc_info.value)


@pytest.mark.asyncio
@patch("litellm.acompletion")
async def test_litellm_reviewer_success(mock_acompletion, empty_memory):
    mock_choice = MagicMock()
    mock_choice.message.parsed = SemanticReview(findings=[])
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_acompletion.return_value = mock_response

    reviewer = LiteLLMMemoryReviewer(model="openai/gpt-4")
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-dummy"}):
        result = await reviewer.review("User: Hello", empty_memory)

    assert isinstance(result, SemanticReview)
    assert len(result.findings) == 0


@pytest.mark.asyncio
@patch("litellm.acompletion")
async def test_litellm_reviewer_timeout_handling(mock_acompletion, empty_memory):
    import litellm
    mock_acompletion.side_effect = litellm.exceptions.Timeout(
        message="Timeout error", model="openai/gpt-4", llm_provider="openai"
    )

    reviewer = LiteLLMMemoryReviewer(model="openai/gpt-4")
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-dummy"}):
        with pytest.raises(ReviewerError) as exc_info:
            await reviewer.review("User: Hello", empty_memory)
    assert "LiteLLM reviewer timed out" in str(exc_info.value)


@pytest.mark.asyncio
@patch("litellm.acompletion")
async def test_litellm_reviewer_auth_error_handling(mock_acompletion, empty_memory):
    import litellm
    mock_acompletion.side_effect = litellm.exceptions.AuthenticationError(
        message="Invalid Key", model="openai/gpt-4", llm_provider="openai"
    )

    reviewer = LiteLLMMemoryReviewer(model="openai/gpt-4")
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-dummy"}):
        with pytest.raises(ReviewerError) as exc_info:
            await reviewer.review("User: Hello", empty_memory)
    assert "Authentication failed with reviewer" in str(exc_info.value)
