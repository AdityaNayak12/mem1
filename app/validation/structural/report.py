from app.validation.structural.models import ValidationResult, IssueLevel


def format_report(result: ValidationResult) -> str:
    """Formats a ValidationResult into a human-readable string report."""
    if result.valid and not result.issues:
        return "Validation Successful"

    lines = []
    
    if result.valid:
        lines.append("Validation Successful")
    else:
        lines.append("Validation Failed")

    errors = [i for i in result.issues if i.level == IssueLevel.ERROR]
    warnings = [i for i in result.issues if i.level == IssueLevel.WARNING]

    if errors:
        lines.append("")
        lines.append("Errors")
        for error in errors:
            lines.append("")
            lines.append(f"• {error.code}")
            if error.location:
                lines.append(f"  Location: {error.location}")
            if error.message:
                lines.append(f"  Message: {error.message}")

    if warnings:
        lines.append("")
        lines.append("Warnings")
        for warning in warnings:
            lines.append("")
            lines.append(f"• {warning.code}")
            if warning.location:
                lines.append(f"  Location: {warning.location}")
            if warning.message:
                lines.append(f"  Message: {warning.message}")

    return "\n".join(lines).strip()
