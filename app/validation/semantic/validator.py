from app.schemas.memory_ir import MemoryIR
from app.validation.structural.models import ValidationResult, ValidationIssue, IssueLevel
from app.validation.semantic.reviewer import MemoryReviewer
from app.validation.semantic.exceptions import SemanticValidationError


class SemanticValidator:

    def __init__(self, reviewer: MemoryReviewer):
        self.reviewer = reviewer

    async def validate(
        self,
        conversation: str,
        memory: MemoryIR,
    ) -> ValidationResult:

        try:
            # Execute the LLM review
            review_result = await self.reviewer.review(conversation, memory)
            
            issues = []
            for finding in review_result.findings:
                # Map reviewer severity to validation IssueLevel
                level = (
                    IssueLevel.ERROR 
                    if finding.severity == "ERROR" 
                    else IssueLevel.WARNING
                )
                
                # Format finding type to code (e.g., 'unsupported_entity' -> 'SEMANTIC_UNSUPPORTED_ENTITY')
                code = f"SEMANTIC_{finding.finding_type.upper()}"
                
                # Combine explanation and recommendation into message
                message = f"{finding.explanation} Recommendation: {finding.recommendation}"

                issues.append(
                    ValidationIssue(
                        level=level,
                        code=code,
                        message=message,
                        location=finding.location,
                    )
                )

            # Valid only if there are no ERROR level issues
            valid = not any(i.level == IssueLevel.ERROR for i in issues)
            return ValidationResult(valid=valid, issues=issues)

        except SemanticValidationError as e:
            # Handle timeout, provider failure, malformed response, or invalid schema
            # by converting them into an ERROR validation issue
            failure_issue = ValidationIssue(
                level=IssueLevel.ERROR,
                code="SEMANTIC_REVIEW_FAILURE",
                message=f"Semantic review failed due to a system error: {str(e)}",
                location=None
            )
            return ValidationResult(valid=False, issues=[failure_issue])
        except Exception as e:
            # Safety catch-all to prevent crashes
            failure_issue = ValidationIssue(
                level=IssueLevel.ERROR,
                code="SEMANTIC_UNEXPECTED_FAILURE",
                message=f"Semantic validator encountered an unexpected failure: {str(e)}",
                location=None
            )
            return ValidationResult(valid=False, issues=[failure_issue])
