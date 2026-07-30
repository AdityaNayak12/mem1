import os
import time
from typing import Any, Protocol
from pydantic import ValidationError

import litellm

from app.core.logging import logger
from app.schemas.memory_ir import MemoryIR
from app.validation.semantic.models import SemanticReview
from app.validation.semantic.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from app.validation.semantic.exceptions import ReviewerError, InvalidReviewResponseError


class MemoryReviewer(Protocol):
    async def review(
        self,
        conversation: str,
        memory: MemoryIR,
    ) -> SemanticReview:
        ...


class LiteLLMMemoryReviewer(MemoryReviewer):

    def __init__(self, model: str | None = None):

        self.model = model or os.getenv("MEM1_VALIDATOR_MODEL", "gpt-4o-mini")
        self.provider = "unknown"
        if self.model:
            parts = self.model.split("/")
            if len(parts) > 1:
                self.provider = parts[0]
            else:
                self.provider = "openai"

        logger.info(f"Initialized LiteLLMMemoryReviewer with model: {self.model} (provider: {self.provider})")

    async def review(
        self,
        conversation: str,
        memory: MemoryIR,
    ) -> SemanticReview:
        if not self.model:
            logger.error("LiteLLMMemoryReviewer call failed: Model name is missing")
            raise ReviewerError("Model name is missing or not configured. Set MEM1_VALIDATOR_MODEL.")

        # Key checks
        if self.provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
            logger.error("LiteLLMMemoryReviewer call failed: Missing OPENROUTER_API_KEY")
            raise ReviewerError("Authentication failed: Missing OPENROUTER_API_KEY in environment.")
        elif self.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            logger.error("LiteLLMMemoryReviewer call failed: Missing OPENAI_API_KEY")
            raise ReviewerError("Authentication failed: Missing OPENAI_API_KEY in environment.")

        # Serialize memory IR to JSON format for review prompt
        memory_json_str = memory.model_dump_json(indent=2)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            conversation=conversation,
            memory_ir_json=memory_json_str
        )

        start_time = time.perf_counter()
        logger.info(f"Sending semantic review request using model {self.model}")

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=SemanticReview,
            )

            duration = time.perf_counter() - start_time
            logger.info(f"Semantic review request succeeded in {duration:.3f} seconds")

            if not response or not response.choices:
                raise InvalidReviewResponseError("Empty response from LiteLLM reviewer")

            message = response.choices[0].message

            # Handle Pydantic response parsing from LiteLLM structured outputs
            if hasattr(message, "parsed") and message.parsed is not None:
                if isinstance(message.parsed, SemanticReview):
                    return message.parsed
                try:
                    return SemanticReview.model_validate(message.parsed)
                except ValidationError as e:
                    raise InvalidReviewResponseError(f"Pydantic validation failed on parsed response: {e}") from e

            # Fallback to parsing from content string
            if hasattr(message, "content") and message.content is not None:
                content_str = message.content.strip()
                # Clean potential markdown wrappers
                if content_str.startswith("```"):
                    lines = content_str.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    content_str = "\n".join(lines).strip()
                try:
                    return SemanticReview.model_validate_json(content_str)
                except Exception as e:
                    raise InvalidReviewResponseError(f"Failed to parse content string to SemanticReview: {e}") from e

            raise InvalidReviewResponseError("No content or parsed object returned by reviewer")

        except litellm.exceptions.Timeout as e:
            duration = time.perf_counter() - start_time
            logger.error(f"LiteLLM reviewer timed out after {duration:.3f}s: {e}")
            raise ReviewerError(f"LiteLLM reviewer timed out: {e}") from e
        except litellm.exceptions.AuthenticationError as e:
            logger.error(f"LiteLLM reviewer auth failed: {e}")
            raise ReviewerError(f"Authentication failed with reviewer: {e}") from e
        except litellm.exceptions.APIConnectionError as e:
            logger.error(f"LiteLLM reviewer connection failed: {e}")
            raise ReviewerError(f"Reviewer provider unavailable (connection error): {e}") from e
        except litellm.exceptions.APIError as e:
            logger.error(f"LiteLLM reviewer API error: {e}")
            raise ReviewerError(f"Reviewer API error: {e}") from e
        except Exception as e:
            logger.error(f"LiteLLM reviewer general error: {e}")
            raise ReviewerError(f"Reviewer provider error: {e}") from e
