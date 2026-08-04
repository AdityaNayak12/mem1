from enum import Enum
from pydantic import BaseModel, Field

# Re-use IssueLevel from structural validation models
from app.validation.structural.models import IssueLevel


class FindingType(str, Enum):
    UNSUPPORTED_ENTITY = "UNSUPPORTED_ENTITY"
    UNSUPPORTED_RELATIONSHIP = "UNSUPPORTED_RELATIONSHIP"
    UNSUPPORTED_EVENT = "UNSUPPORTED_EVENT"
    HALLUCINATION = "HALLUCINATION"
    MISSING_MEMORY = "MISSING_MEMORY"


class Recommendation(str, Enum):
    KEEP = "KEEP"
    REMOVE = "REMOVE"
    MODIFY = "MODIFY"
    ADD = "ADD"
    REVIEW = "REVIEW"


class ReviewFinding(BaseModel):
    finding_type: FindingType = Field(
        ...,
        description="The categorised type of the finding."
    )
    severity: IssueLevel = Field(
        ...,
        description="The validation issue level severity."
    )
    location: str = Field(
        ...,
        description="Location identifier (e.g., 'entities[0]', 'relationships[3]')."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this finding, constrained between 0.0 and 1.0."
    )
    explanation: str = Field(
        ...,
        description="Detailed explanation of the issue."
    )
    evidence: str = Field(
        ...,
        description="Evidence text snippet from the conversation backing this finding."
    )
    recommendation: Recommendation = Field(
        ...,
        description="Machine-readable action recommendation."
    )
    suggested_fix: str | None = Field(
        None,
        description="Optional text fix suggesting how to resolve the finding."
    )


class SemanticReview(BaseModel):
    findings: list[ReviewFinding] = Field(
        default_factory=list,
        description="List of specific review findings."
    )
    total_memories: int = Field(
        ...,
        ge=0,
        description="Total number of memories evaluated."
    )
    grounded_memories: int = Field(
        ...,
        ge=0,
        description="Total number of correctly grounded memories."
    )
    unsupported_memories: int = Field(
        ...,
        ge=0,
        description="Total number of unsupported memories."
    )
    missing_memories: int = Field(
        ...,
        ge=0,
        description="Total number of missing memories."
    )
    extraction_quality: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall quality of extraction from 0.0 to 1.0."
    )
    summary: str = Field(
        ...,
        description="Narrative summary describing the quality of the extraction."
    )
