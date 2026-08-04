from app.schemas.memory_ir import MemoryIR
from app.validation.structural.models import ValidationResult, ValidationIssue, IssueLevel
from app.validation.semantic.reviewer import MemoryReviewer
from app.validation.semantic.models import FindingType
from app.validation.semantic.exceptions import (
    SemanticReviewError,
    ReviewerTimeout,
    InvalidReviewerResponse,
)


class SemanticValidator:
    """Validator that performs semantic correctness evaluation on extracted MemoryIR.

    Delegates the reasoning and analysis to a MemoryReviewer, then converts findings
    into formal ValidationIssues deterministically.
    """

    # Mapping of reviewer finding types to application validation error codes
    CODE_MAPPING = {
        FindingType.UNSUPPORTED_ENTITY: "ENTITY_NOT_GROUNDED",
        FindingType.UNSUPPORTED_RELATIONSHIP: "RELATIONSHIP_NOT_GROUNDED",
        FindingType.UNSUPPORTED_EVENT: "EVENT_NOT_GROUNDED",
        FindingType.HALLUCINATION: "SEMANTIC_HALLUCINATION",
        FindingType.MISSING_MEMORY: "SEMANTIC_INCOMPLETE",
    }

    def __init__(self, reviewer: MemoryReviewer):
        """Initializes the SemanticValidator with a dependency-injected reviewer.

        Args:
            reviewer: A reviewer instance conforming to the MemoryReviewer protocol.
        """
        self.reviewer = reviewer

    async def validate(
        self,
        conversation: str,
        memory: MemoryIR,
    ) -> ValidationResult:
        """Evaluates whether the MemoryIR represents the conversation faithfully.

        Args:
            conversation: The original conversation transcript text.
            memory: The extracted MemoryIR model object.

        Returns:
            ValidationResult: The validation outcome.
        """
        try:
            # 1. Call the MemoryReviewer
            review_result = await self.reviewer.review(conversation, memory)
            
            issues = []
            
            # 2. Convert ReviewFindings into ValidationIssues
            for finding in review_result.findings:
                # Map FindingType to strict validation code
                code = self.CODE_MAPPING.get(finding.finding_type, "SEMANTIC_UNKNOWN_FINDING")
                
                # Build rich narrative message from details
                message_parts = [finding.explanation]
                
                # Recommendation enum name
                rec_str = finding.recommendation.value
                message_parts.append(f"Recommendation: {rec_str}")
                
                if finding.suggested_fix:
                    message_parts.append(f"Suggested Fix: {finding.suggested_fix}")
                
                message_parts.append(f"(Reviewer Confidence: {finding.confidence:.2f})")
                message = " ".join(message_parts)

                issues.append(
                    ValidationIssue(
                        level=finding.severity,  # Maps directly to IssueLevel (ERROR/WARNING)
                        code=code,
                        message=message,
                        location=finding.location,
                    )
                )

            # 3. Determine overall validity (True only if no ERROR level issues exist)
            valid = not any(i.level == IssueLevel.ERROR for i in issues)
            return ValidationResult(valid=valid, issues=issues)

        except ReviewerTimeout as e:
            # Map timeout exception into a formal ERROR validation issue
            timeout_issue = ValidationIssue(
                level=IssueLevel.ERROR,
                code="SEMANTIC_TIMEOUT",
                message=f"Semantic review timed out: {str(e)}",
                location=None
            )
            return ValidationResult(valid=False, issues=[timeout_issue])
            
        except InvalidReviewerResponse as e:
            # Map invalid response exception into a formal ERROR validation issue
            invalid_issue = ValidationIssue(
                level=IssueLevel.ERROR,
                code="SEMANTIC_INVALID_RESPONSE",
                message=f"Reviewer returned an invalid or malformed response: {str(e)}",
                location=None
            )
            return ValidationResult(valid=False, issues=[invalid_issue])
            
        except SemanticReviewError as e:
            # Map general review error into a formal ERROR validation issue
            error_issue = ValidationIssue(
                level=IssueLevel.ERROR,
                code="SEMANTIC_REVIEW_FAILURE",
                message=f"Semantic reviewer failure occurred: {str(e)}",
                location=None
            )
            return ValidationResult(valid=False, issues=[error_issue])
            
        except Exception as e:
            # Safety catch-all to prevent crashes
            unexpected_issue = ValidationIssue(
                level=IssueLevel.ERROR,
                code="SEMANTIC_UNEXPECTED_FAILURE",
                message=f"Semantic validator encountered an unexpected failure: {str(e)}",
                location=None
            )
            return ValidationResult(valid=False, issues=[unexpected_issue])
