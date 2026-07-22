from enum import Enum
from pydantic import BaseModel, Field


class IssueLevel(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


class ValidationIssue(BaseModel):
    level: IssueLevel
    code: str
    message: str
    location: str | None = None


class ValidationResult(BaseModel):
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)