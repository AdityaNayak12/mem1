from app.schemas.memory_ir import MemoryIR
from app.validation.structural.models import ValidationIssue, IssueLevel


def validate_entities(memory: MemoryIR) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seen_ids = set()

    for idx, entity in enumerate(memory.entities):
        location = f"entities[{idx}]"

        # 1. Duplicate entity ID check
        if entity.id in seen_ids:
            issues.append(
                ValidationIssue(
                    level=IssueLevel.ERROR,
                    code="ENTITY_DUPLICATE_ID",
                    message=f"Duplicate entity ID found: {entity.id}",
                    location=location,
                )
            )
        else:
            seen_ids.add(entity.id)

        # 2. Entity name is not empty or whitespace
        if not entity.name or not entity.name.strip():
            issues.append(
                ValidationIssue(
                    level=IssueLevel.ERROR,
                    code="ENTITY_NAME_EMPTY",
                    message="Entity name is empty or contains only whitespace",
                    location=location,
                )
            )

        # 3. Entity type is missing warning
        if not entity.type or not entity.type.strip():
            issues.append(
                ValidationIssue(
                    level=IssueLevel.WARNING,
                    code="ENTITY_TYPE_MISSING",
                    message="Entity type is missing or empty",
                    location=location,
                )
            )

        # 4. Aliases contain no duplicates
        if entity.aliases:
            if len(entity.aliases) != len(set(entity.aliases)):
                issues.append(
                    ValidationIssue(
                        level=IssueLevel.ERROR,
                        code="ENTITY_DUPLICATE_ALIAS",
                        message="Entity aliases contain duplicate values",
                        location=location,
                    )
                )


        # 5. Evidence checks
        if not entity.evidence:
            issues.append(
                ValidationIssue(
                    level=IssueLevel.ERROR,
                    code="EVIDENCE_MISSING",
                    message="Evidence is missing for entity",
                    location=location,
                )
            )
        else:
            if not entity.evidence.message_id or not entity.evidence.message_id.strip():
                issues.append(
                    ValidationIssue(
                        level=IssueLevel.ERROR,
                        code="EVIDENCE_EMPTY_MESSAGE_ID",
                        message="Evidence message_id is empty or whitespace",
                        location=location,
                    )
                )
            if not entity.evidence.text or not entity.evidence.text.strip():
                issues.append(
                    ValidationIssue(
                        level=IssueLevel.ERROR,
                        code="EVIDENCE_EMPTY_TEXT",
                        message="Evidence text is empty or whitespace",
                        location=location,
                    )
                )

    return issues


def validate_relationships(memory: MemoryIR) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seen_ids = set()
    entity_ids = {e.id for e in memory.entities}

    for idx, rel in enumerate(memory.relationships):
        location = f"relationships[{idx}]"

        # 1. Duplicate relationship ID check
        if rel.id in seen_ids:
            issues.append(
                ValidationIssue(
                    level=IssueLevel.ERROR,
                    code="RELATIONSHIP_DUPLICATE_ID",
                    message=f"Duplicate relationship ID found: {rel.id}",
                    location=location,
                )
            )
        else:
            seen_ids.add(rel.id)

        # 2. Subject references existing entity ID
        if rel.subject not in entity_ids:
            issues.append(
                ValidationIssue(
                    level=IssueLevel.ERROR,
                    code="RELATIONSHIP_INVALID_SUBJECT",
                    message=f"Relationship subject ID {rel.subject} does not reference an existing entity",
                    location=location,
                )
            )

        # 3. Object references existing entity ID
        if rel.object not in entity_ids:
            issues.append(
                ValidationIssue(
                    level=IssueLevel.ERROR,
                    code="RELATIONSHIP_INVALID_OBJECT",
                    message=f"Relationship object ID {rel.object} does not reference an existing entity",
                    location=location,
                )
            )

        # 4. Predicate checks
        if not rel.predicate or not rel.predicate.strip():
            issues.append(
                ValidationIssue(
                    level=IssueLevel.ERROR,
                    code="RELATIONSHIP_EMPTY_PREDICATE",
                    message="Relationship predicate is empty or whitespace",
                    location=location,
                )
            )


        # 6. Evidence checks
        if not rel.evidence:
            issues.append(
                ValidationIssue(
                    level=IssueLevel.ERROR,
                    code="EVIDENCE_MISSING",
                    message="Evidence is missing for relationship",
                    location=location,
                )
            )
        else:
            if not rel.evidence.message_id or not rel.evidence.message_id.strip():
                issues.append(
                    ValidationIssue(
                        level=IssueLevel.ERROR,
                        code="EVIDENCE_EMPTY_MESSAGE_ID",
                        message="Evidence message_id is empty or whitespace",
                        location=location,
                    )
                )
            if not rel.evidence.text or not rel.evidence.text.strip():
                issues.append(
                    ValidationIssue(
                        level=IssueLevel.ERROR,
                        code="EVIDENCE_EMPTY_TEXT",
                        message="Evidence text is empty or whitespace",
                        location=location,
                    )
                )

    return issues


def validate_events(memory: MemoryIR) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seen_ids = set()
    entity_ids = {e.id for e in memory.entities}

    for idx, event in enumerate(memory.events):
        location = f"events[{idx}]"

        # 1. Duplicate event ID check
        if event.id in seen_ids:
            issues.append(
                ValidationIssue(
                    level=IssueLevel.ERROR,
                    code="EVENT_DUPLICATE_ID",
                    message=f"Duplicate event ID found: {event.id}",
                    location=location,
                )
            )
        else:
            seen_ids.add(event.id)

        # 2. Participants reference existing entity ID
        for p_idx, participant in enumerate(event.participants):
            if participant not in entity_ids:
                issues.append(
                    ValidationIssue(
                        level=IssueLevel.ERROR,
                        code="EVENT_INVALID_PARTICIPANT",
                        message=f"Event participant ID {participant} at index {p_idx} does not reference an existing entity",
                        location=location,
                    )
                )


        # 4. Evidence checks
        if not event.evidence:
            issues.append(
                ValidationIssue(
                    level=IssueLevel.ERROR,
                    code="EVIDENCE_MISSING",
                    message="Evidence is missing for event",
                    location=location,
                )
            )
        else:
            if not event.evidence.message_id or not event.evidence.message_id.strip():
                issues.append(
                    ValidationIssue(
                        level=IssueLevel.ERROR,
                        code="EVIDENCE_EMPTY_MESSAGE_ID",
                        message="Evidence message_id is empty or whitespace",
                        location=location,
                    )
                )
            if not event.evidence.text or not event.evidence.text.strip():
                issues.append(
                    ValidationIssue(
                        level=IssueLevel.ERROR,
                        code="EVIDENCE_EMPTY_TEXT",
                        message="Evidence text is empty or whitespace",
                        location=location,
                    )
                )

    return issues
