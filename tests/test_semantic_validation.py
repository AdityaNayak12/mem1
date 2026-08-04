import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.memory_ir import MemoryIR
from app.validation.structural.models import IssueLevel
from app.validation.semantic import (
    SemanticValidator,
    MemoryReviewer,
    LiteLLMMemoryReviewer,
    SemanticReview,
    ReviewFinding,
    FindingType,
    Recommendation,
    SemanticReviewError,
    ReviewerTimeout,
    InvalidReviewerResponse,
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


@pytest.fixture
def base_review_stats():
    """Helper to provide baseline statistics for SemanticReview models."""
    return {
        "total_memories": 5,
        "grounded_memories": 5,
        "unsupported_memories": 0,
        "missing_memories": 0,
        "extraction_quality": 1.0,
        "summary": "Extraction is very good.",
    }


@pytest.mark.asyncio
async def test_supported_extraction(validator, mock_reviewer, empty_memory, base_review_stats):
    # Setup mock to return a review with no findings
    mock_reviewer.review_mock.return_value = SemanticReview(
        findings=[],
        **base_review_stats
    )

    result = await validator.validate("User: Hello", empty_memory)

    assert result.valid is True
    assert len(result.issues) == 0
    mock_reviewer.review_mock.assert_called_once_with("User: Hello", empty_memory)


@pytest.mark.asyncio
async def test_unsupported_entity(validator, mock_reviewer, empty_memory, base_review_stats):
    stats = base_review_stats.copy()
    stats["unsupported_memories"] = 1
    stats["extraction_quality"] = 0.8

    mock_reviewer.review_mock.return_value = SemanticReview(
        findings=[
            ReviewFinding(
                finding_type=FindingType.UNSUPPORTED_ENTITY,
                severity=IssueLevel.ERROR,
                location="entities[0]",
                confidence=0.9,
                explanation="Entity Bob was not mentioned.",
                evidence="N/A",
                recommendation=Recommendation.REMOVE,
                suggested_fix="Delete entity Bob."
            )
        ],
        **stats
    )

    result = await validator.validate("User: Hello", empty_memory)

    assert result.valid is False
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.level == IssueLevel.ERROR
    assert issue.code == "ENTITY_NOT_GROUNDED"
    assert issue.location == "entities[0]"
    assert "Entity Bob was not mentioned." in issue.message
    assert "Recommendation: REMOVE" in issue.message
    assert "Suggested Fix: Delete entity Bob." in issue.message
    assert "Reviewer Confidence: 0.90" in issue.message


@pytest.mark.asyncio
async def test_unsupported_relationship(validator, mock_reviewer, empty_memory, base_review_stats):
    stats = base_review_stats.copy()
    stats["unsupported_memories"] = 1
    stats["extraction_quality"] = 0.8

    mock_reviewer.review_mock.return_value = SemanticReview(
        findings=[
            ReviewFinding(
                finding_type=FindingType.UNSUPPORTED_RELATIONSHIP,
                severity=IssueLevel.ERROR,
                location="relationships[0]",
                confidence=0.85,
                explanation="Alice doesn't work at Google.",
                evidence="I work at Apple.",
                recommendation=Recommendation.MODIFY,
                suggested_fix="Change target to Apple."
            )
        ],
        **stats
    )

    result = await validator.validate("User: Hello", empty_memory)

    assert result.valid is False
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.level == IssueLevel.ERROR
    assert issue.code == "RELATIONSHIP_NOT_GROUNDED"
    assert issue.location == "relationships[0]"
    assert "Alice doesn't work at Google." in issue.message
    assert "Recommendation: MODIFY" in issue.message
    assert "Reviewer Confidence: 0.85" in issue.message


@pytest.mark.asyncio
async def test_unsupported_event(validator, mock_reviewer, empty_memory, base_review_stats):
    stats = base_review_stats.copy()
    stats["unsupported_memories"] = 1
    stats["extraction_quality"] = 0.8

    mock_reviewer.review_mock.return_value = SemanticReview(
        findings=[
            ReviewFinding(
                finding_type=FindingType.UNSUPPORTED_EVENT,
                severity=IssueLevel.ERROR,
                location="events[0]",
                confidence=0.95,
                explanation="No acquisition took place.",
                evidence="N/A",
                recommendation=Recommendation.REMOVE,
                suggested_fix="Remove event."
            )
        ],
        **stats
    )

    result = await validator.validate("User: Hello", empty_memory)

    assert result.valid is False
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.level == IssueLevel.ERROR
    assert issue.code == "EVENT_NOT_GROUNDED"
    assert issue.location == "events[0]"


@pytest.mark.asyncio
async def test_hallucination(validator, mock_reviewer, empty_memory, base_review_stats):
    stats = base_review_stats.copy()
    stats["unsupported_memories"] = 1
    stats["extraction_quality"] = 0.8

    mock_reviewer.review_mock.return_value = SemanticReview(
        findings=[
            ReviewFinding(
                finding_type=FindingType.HALLUCINATION,
                severity=IssueLevel.ERROR,
                location="entities[1]",
                confidence=1.0,
                explanation="Invented details about project status.",
                evidence="N/A",
                recommendation=Recommendation.REMOVE,
                suggested_fix="Remove property status."
            )
        ],
        **stats
    )

    result = await validator.validate("User: Hello", empty_memory)

    assert result.valid is False
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.level == IssueLevel.ERROR
    assert issue.code == "SEMANTIC_HALLUCINATION"
    assert issue.location == "entities[1]"


@pytest.mark.asyncio
async def test_missing_memory(validator, mock_reviewer, empty_memory, base_review_stats):
    stats = base_review_stats.copy()
    stats["missing_memories"] = 1
    stats["extraction_quality"] = 0.9

    mock_reviewer.review_mock.return_value = SemanticReview(
        findings=[
            ReviewFinding(
                finding_type=FindingType.MISSING_MEMORY,
                severity=IssueLevel.WARNING,
                location="general",
                confidence=0.8,
                explanation="Omitted Python programming language.",
                evidence="I code in Python.",
                recommendation=Recommendation.ADD,
                suggested_fix="Add Python entity."
            )
        ],
        **stats
    )

    result = await validator.validate("User: Hello", empty_memory)

    # Warning issues should keep result.valid as True
    assert result.valid is True
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.level == IssueLevel.WARNING
    assert issue.code == "SEMANTIC_INCOMPLETE"
    assert issue.location == "general"


@pytest.mark.asyncio
async def test_invalid_review_exception(validator, mock_reviewer, empty_memory):
    mock_reviewer.review_mock.side_effect = InvalidReviewerResponse("Response missing findings list")

    result = await validator.validate("User: Hello", empty_memory)

    assert result.valid is False
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.level == IssueLevel.ERROR
    assert issue.code == "SEMANTIC_INVALID_RESPONSE"
    assert "Response missing findings list" in issue.message


@pytest.mark.asyncio
async def test_timeout_exception(validator, mock_reviewer, empty_memory):
    mock_reviewer.review_mock.side_effect = ReviewerTimeout("LLM request timed out after 30s")

    result = await validator.validate("User: Hello", empty_memory)

    assert result.valid is False
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.level == IssueLevel.ERROR
    assert issue.code == "SEMANTIC_TIMEOUT"
    assert "LLM request timed out" in issue.message


@pytest.mark.asyncio
async def test_provider_failure_exception(validator, mock_reviewer, empty_memory):
    mock_reviewer.review_mock.side_effect = SemanticReviewError("API connection failure")

    result = await validator.validate("User: Hello", empty_memory)

    assert result.valid is False
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.level == IssueLevel.ERROR
    assert issue.code == "SEMANTIC_REVIEW_FAILURE"
    assert "Semantic reviewer failure occurred" in issue.message


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
        with pytest.raises(SemanticReviewError) as exc_info:
            await reviewer.review("User: Hello", empty_memory)
        assert "Missing OPENROUTER_API_KEY" in str(exc_info.value)


@pytest.mark.asyncio
@patch("litellm.acompletion")
async def test_litellm_reviewer_success(mock_acompletion, empty_memory, base_review_stats):
    mock_choice = MagicMock()
    mock_choice.message.parsed = SemanticReview(
        findings=[],
        **base_review_stats
    )
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_acompletion.return_value = mock_response

    reviewer = LiteLLMMemoryReviewer(model="openai/gpt-4")
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-dummy"}):
        result = await reviewer.review("User: Hello", empty_memory)

    assert isinstance(result, SemanticReview)
    assert len(result.findings) == 0
    assert result.extraction_quality == 1.0


@pytest.mark.asyncio
@patch("litellm.acompletion")
async def test_litellm_reviewer_timeout_handling(mock_acompletion, empty_memory):
    import litellm
    mock_acompletion.side_effect = litellm.exceptions.Timeout(
        message="Timeout error", model="openai/gpt-4", llm_provider="openai"
    )

    reviewer = LiteLLMMemoryReviewer(model="openai/gpt-4")
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-dummy"}):
        with pytest.raises(ReviewerTimeout) as exc_info:
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
        with pytest.raises(SemanticReviewError) as exc_info:
            await reviewer.review("User: Hello", empty_memory)
    assert "Authentication failed with reviewer" in str(exc_info.value)
