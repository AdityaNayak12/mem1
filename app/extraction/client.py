import abc
import os
import time
from typing import Any, Type
from pydantic import BaseModel

# Import litellm only in this client implementation file
import litellm

from app.core.logging import logger
from app.extraction.exceptions import ProviderError, InvalidResponseError


class LLMClient(abc.ABC):
    """Abstract base class for LLM client providers."""

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        response_model: Type[BaseModel] | None = None,
    ) -> Any:
        """Call the LLM provider asynchronously and return structured or text response."""
        pass


class LiteLLMClient(LLMClient):
    """LiteLLM implementation of the LLMClient."""

    def __init__(self, model: str | None = None):
        """Initializes LiteLLMClient.

        Args:
            model: Optional model override. Reads MEM1_EXTRACTION_MODEL environment variable.
        """
        self.model = model or os.getenv("MEM1_EXTRACTION_MODEL")
        # Extract provider from model string (e.g. "openrouter/google/gemini-2.5-flash-lite" -> "openrouter")
        self.provider = "unknown"
        if self.model:
            parts = self.model.split("/")
            if len(parts) > 1:
                self.provider = parts[0]
            else:
                self.provider = "openai"  # default provider for simple model names in litellm

        logger.info(f"Initialized LiteLLMClient with model: {self.model} (provider: {self.provider})")

    async def complete(
        self,
        messages: list[dict[str, Any]],
        response_model: Type[BaseModel] | None = None,
    ) -> Any:
        """Invokes LiteLLM asynchronously to complete the chat prompt.

        Args:
            messages: List of chat message dictionaries.
            response_model: Optional Pydantic class to enforce structured schema validation.

        Returns:
            The parsed Pydantic object if response_model is provided, otherwise string content.

        Raises:
            ProviderError: For API keys issues, timeouts, connection issues, or other provider errors.
            InvalidResponseError: For bad response formats or parsing errors.
        """
        # Validate model presence
        if not self.model:
            logger.error("LiteLLMClient call failed: Model name is missing")
            raise ProviderError("Model name is missing or not configured. Set MEM1_EXTRACTION_MODEL.")

        # Validate API keys based on provider
        if self.provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
            logger.error("LiteLLMClient call failed: Missing OPENROUTER_API_KEY")
            raise ProviderError("Authentication failed: Missing OPENROUTER_API_KEY in environment.")
        elif self.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            # Check OpenAI key if it defaults/maps to OpenAI
            logger.error("LiteLLMClient call failed: Missing OPENAI_API_KEY")
            raise ProviderError("Authentication failed: Missing OPENAI_API_KEY in environment.")

        start_time = time.perf_counter()
        logger.info(f"Sending completion request using model {self.model}")

        try:
            # LiteLLM async completion call
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                response_format=response_model,
            )

            duration = time.perf_counter() - start_time
            logger.info(f"Completion request succeeded in {duration:.3f} seconds")

            if not response or not response.choices:
                raise InvalidResponseError("Empty response structure received from LiteLLM provider")

            message = response.choices[0].message

            # Check if LiteLLM / provider already parsed the model
            if response_model is not None:
                if hasattr(message, "parsed") and message.parsed is not None:
                    return message.parsed

            # Fall back to returning string content to parse downstream
            if hasattr(message, "content") and message.content is not None:
                return message.content

            raise InvalidResponseError("No content or parsed object returned in message choices")

        except litellm.exceptions.Timeout as e:
            duration = time.perf_counter() - start_time
            logger.error(f"LiteLLM request timed out after {duration:.3f}s: {e}")
            raise ProviderError(f"LiteLLM call timed out: {e}") from e
        except litellm.exceptions.AuthenticationError as e:
            logger.error(f"LiteLLM authentication failed: {e}")
            raise ProviderError(f"Authentication failed with provider: {e}") from e
        except litellm.exceptions.APIConnectionError as e:
            logger.error(f"LiteLLM connection failed: {e}")
            raise ProviderError(f"Provider unavailable (connection error): {e}") from e
        except litellm.exceptions.APIError as e:
            logger.error(f"LiteLLM API error: {e}")
            raise ProviderError(f"LLM Provider API error: {e}") from e
        except Exception as e:
            logger.error(f"LiteLLM general error: {e}")
            raise ProviderError(f"LiteLLM provider error: {e}") from e
