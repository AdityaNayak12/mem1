from typing import Literal
from pydantic import BaseModel, Field


class ReviewFinding(BaseModel):
    finding_type: Literal[
        "unsupported_entity",
        "unsupported_relationship",
        "unsupported_event",
        "hallucination",
        "missing_memory",
    ] = Field(
        ...,
        description="The type of semantic validation finding."
    )
    severity: Literal["ERROR", "WARNING"] = Field(
        ...,
        description="The severity level of the finding. Hallucinations and unsupported extractions should be ERROR. Missing memory should be WARNING."
    )
    location: str = Field(
        ...,
        description="Location of the object being validated in standard format (e.g., 'entities[0]', 'relationships[3]', 'events[1]'). For missing_memory, specify a generic section or conversation context."
    )
    explanation: str = Field(
        ...,
        description="Detailed explanation of why this finding is being reported, referencing the conversation."
    )
    recommendation: str = Field(
        ...,
        description="Actionable recommendation on how to address this finding."
    )


class SemanticReview(BaseModel):
    findings: list[ReviewFinding] = Field(
        default_factory=list,
        description="List of review findings comparing the conversation to the extracted MemoryIR."
    )
