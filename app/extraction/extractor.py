from app.schemas.memory_ir import MemoryIR
from app.extraction.client import LLMClient
from app.extraction.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from app.extraction.parser import parse_response


class MemoryExtractor:
    def __init__(self, client: LLMClient):
        self.client = client

    async def extract(self, conversation: str) -> MemoryIR:
        # Guard for empty conversation inputs
        if not conversation or not conversation.strip():
            return MemoryIR(entities=[], relationships=[], events=[])

        # Construct the user prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(conversation=conversation)

        # Call the LLM client using complete with chat messages
        raw_response = await self.client.complete(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_model=MemoryIR,
        )

        # Parse and return validated MemoryIR
        return parse_response(raw_response)
