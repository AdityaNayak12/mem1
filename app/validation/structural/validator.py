from app.schemas.memory_ir import MemoryIR
from app.validation.structural.models import ValidationResult, IssueLevel
from app.validation.structural.rules import (
    validate_entities,
    validate_relationships,
    validate_events,
)


class StructuralValidator:
    RULES = (
        validate_entities,
        validate_relationships,
        validate_events,
    )

    def validate(self, memory: MemoryIR) -> ValidationResult:
        issues = []

        for rule in self.RULES:
            issues.extend(rule(memory))

        valid = not any(
            issue.level == IssueLevel.ERROR
            for issue in issues
        )

        return ValidationResult(valid=valid, issues=issues)