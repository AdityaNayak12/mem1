import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.memory_ir import MemoryIR, Entity, Relationship, Event, Evidence
from app.extraction.client import LLMClient, LiteLLMClient
from app.extraction.extractor import MemoryExtractor
from app.extraction.exceptions import ProviderError, InvalidResponseError
from app.extraction.parser import parse_response


class MockLLMClient(LLMClient):
    """Mock implementation of the LLMClient for testing."""

    def __init__(self):
        self.complete_mock = AsyncMock()

    async def complete(self, messages, response_model=None):
        return await self.complete_mock(messages, response_model)


@pytest.fixture
def mock_client():
    return MockLLMClient()


@pytest.fixture
def extractor(mock_client):
    return MemoryExtractor(client=mock_client)


@pytest.fixture
def valid_memory_ir_dict():
    entity_id = str(uuid.uuid4())
    return {
        "entities": [
            {
                "id": entity_id,
                "name": "FastAPI",
                "type": "Framework",
                "aliases": [],
                "confidence": 0.95,
                "evidence": {"message_id": "msg-1", "text": "We use FastAPI"},
            }
        ],
        "relationships": [],
        "events": [],
    }


@pytest.mark.asyncio
async def test_successful_extraction(extractor, mock_client, valid_memory_ir_dict):
    mock_client.complete_mock.return_value = valid_memory_ir_dict

    conversation = "User: We use FastAPI"
    result = await extractor.extract(conversation)

    assert isinstance(result, MemoryIR)
    assert len(result.entities) == 1
    assert result.entities[0].name == "FastAPI"
    assert result.entities[0].confidence == 0.95
    mock_client.complete_mock.assert_called_once()


@pytest.mark.asyncio
async def test_empty_conversation(extractor, mock_client):
    result = await extractor.extract("")
    assert isinstance(result, MemoryIR)
    assert len(result.entities) == 0
    assert len(result.relationships) == 0
    assert len(result.events) == 0
    mock_client.complete_mock.assert_not_called()

    # Whitespace only
    result = await extractor.extract("   \n   ")
    assert isinstance(result, MemoryIR)
    mock_client.complete_mock.assert_not_called()


@pytest.mark.asyncio
async def test_malformed_json_response(extractor, mock_client):
    # LLM returns invalid JSON string
    mock_client.complete_mock.return_value = '{"entities": [invalid json'

    with pytest.raises(InvalidResponseError) as exc_info:
        await extractor.extract("User: Hello")
    assert "Failed to parse JSON string" in str(exc_info.value)


@pytest.mark.asyncio
async def test_invalid_memory_ir_response(extractor, mock_client):
    # Missing required 'evidence' field for entity
    invalid_data = {
        "entities": [
            {
                "id": str(uuid.uuid4()),
                "name": "FastAPI",
                "confidence": 0.95,
                # evidence is missing
            }
        ]
    }
    mock_client.complete_mock.return_value = invalid_data

    with pytest.raises(InvalidResponseError) as exc_info:
        await extractor.extract("User: Hello")
    assert "Pydantic validation failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_provider_failure(extractor, mock_client):
    mock_client.complete_mock.side_effect = ProviderError("API Connection failed")

    with pytest.raises(ProviderError) as exc_info:
        await extractor.extract("User: Hello")
    assert "API Connection failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_timeout_failure(extractor, mock_client):
    mock_client.complete_mock.side_effect = ProviderError("LiteLLM call timed out")

    with pytest.raises(ProviderError) as exc_info:
        await extractor.extract("User: Hello")
    assert "LiteLLM call timed out" in str(exc_info.value)


def test_parser_failures():
    # Test parser handling None
    with pytest.raises(InvalidResponseError) as exc_info:
        parse_response(None)
    assert "Empty response received from provider" in str(exc_info.value)

    # Test parser handling bad object type
    with pytest.raises(InvalidResponseError) as exc_info:
        parse_response(12345)
    assert "Pydantic validation failed" in str(exc_info.value)

    # Test markdown block stripping
    markdown_json = """```json
{
    "entities": [],
    "relationships": [],
    "events": []
}
```"""
    parsed = parse_response(markdown_json)
    assert isinstance(parsed, MemoryIR)
    assert len(parsed.entities) == 0


# --- LiteLLMClient Specific Tests ---

def test_litellm_client_initialization():
    with patch.dict("os.environ", {"MEM1_EXTRACTION_MODEL": "openrouter/google/gemini-2.5-flash-lite"}):
        client = LiteLLMClient()
        assert client.model == "openrouter/google/gemini-2.5-flash-lite"
        assert client.provider == "openrouter"

    with patch.dict("os.environ", {"MEM1_EXTRACTION_MODEL": "openai/gpt-4"}):
        client = LiteLLMClient()
        assert client.model == "openai/gpt-4"
        assert client.provider == "openai"


@pytest.mark.asyncio
async def test_litellm_client_missing_model():
    client = LiteLLMClient(model=None)
    # Ensure env variable is also not set
    with patch.dict("os.environ", {}, clear=True):
        client.model = None
        with pytest.raises(ProviderError) as exc_info:
            await client.complete([{"role": "user", "content": "test"}])
        assert "Model name is missing" in str(exc_info.value)


@pytest.mark.asyncio
async def test_litellm_client_missing_api_key():
    client = LiteLLMClient(model="openrouter/google/gemini-2.5-flash-lite")
    # Verify openrouter model complains about missing OPENROUTER_API_KEY
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderError) as exc_info:
            await client.complete([{"role": "user", "content": "test"}])
        assert "Missing OPENROUTER_API_KEY" in str(exc_info.value)

    client_openai = LiteLLMClient(model="openai/gpt-4")
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderError) as exc_info:
            await client_openai.complete([{"role": "user", "content": "test"}])
        assert "Missing OPENAI_API_KEY" in str(exc_info.value)


@pytest.mark.asyncio
@patch("litellm.acompletion")
async def test_litellm_client_success(mock_acompletion, valid_memory_ir_dict):
    # Setup mock response from LiteLLM
    mock_choice = MagicMock()
    mock_choice.message.parsed = MemoryIR.model_validate(valid_memory_ir_dict)
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_acompletion.return_value = mock_response

    client = LiteLLMClient(model="openai/gpt-4")
    
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-dummy"}):
        result = await client.complete(
            messages=[{"role": "user", "content": "hello"}],
            response_model=MemoryIR
        )

    assert isinstance(result, MemoryIR)
    assert len(result.entities) == 1
    assert result.entities[0].name == "FastAPI"


@pytest.mark.asyncio
@patch("litellm.acompletion")
async def test_litellm_client_timeout_handling(mock_acompletion):
    import litellm
    # Simulate a timeout exception from LiteLLM
    mock_acompletion.side_effect = litellm.exceptions.Timeout(
        message="Timeout error", model="openai/gpt-4", llm_provider="openai"
    )

    client = LiteLLMClient(model="openai/gpt-4")
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-dummy"}):
        with pytest.raises(ProviderError) as exc_info:
            await client.complete([{"role": "user", "content": "test"}], response_model=MemoryIR)
    assert "LiteLLM call timed out" in str(exc_info.value)


@pytest.mark.asyncio
@patch("litellm.acompletion")
async def test_litellm_client_auth_error_handling(mock_acompletion):
    import litellm
    # Simulate an auth error from LiteLLM
    mock_acompletion.side_effect = litellm.exceptions.AuthenticationError(
        message="Invalid API Key", model="openai/gpt-4", llm_provider="openai"
    )

    client = LiteLLMClient(model="openai/gpt-4")
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-dummy"}):
        with pytest.raises(ProviderError) as exc_info:
            await client.complete([{"role": "user", "content": "test"}], response_model=MemoryIR)
    assert "Authentication failed with provider" in str(exc_info.value)


@pytest.mark.asyncio
@patch("litellm.acompletion")
async def test_litellm_client_connection_error_handling(mock_acompletion):
    import litellm
    mock_acompletion.side_effect = litellm.exceptions.APIConnectionError(
        message="Connection failed", model="openai/gpt-4", llm_provider="openai"
    )

    client = LiteLLMClient(model="openai/gpt-4")
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-dummy"}):
        with pytest.raises(ProviderError) as exc_info:
            await client.complete([{"role": "user", "content": "test"}], response_model=MemoryIR)
    assert "Provider unavailable (connection error)" in str(exc_info.value)


@pytest.mark.asyncio
@patch("litellm.acompletion")
async def test_litellm_client_general_error_handling(mock_acompletion):
    mock_acompletion.side_effect = RuntimeError("Something went wrong")

    client = LiteLLMClient(model="openai/gpt-4")
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-dummy"}):
        with pytest.raises(ProviderError) as exc_info:
            await client.complete([{"role": "user", "content": "test"}], response_model=MemoryIR)
    assert "LiteLLM provider error" in str(exc_info.value)
