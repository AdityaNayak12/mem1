import uuid
from datetime import datetime
import pytest

from app.schemas.memory_ir import MemoryIR, Entity, Relationship, Event, Evidence
from app.validation import StructuralValidator, IssueLevel
from app.validation.report import format_report


@pytest.fixture
def base_evidence():
    return Evidence(message_id="msg-1", text="evidence text")


@pytest.fixture
def validator():
    return StructuralValidator()


def test_valid_memory_ir(validator, base_evidence):
    entity1_id = uuid.uuid4()
    entity2_id = uuid.uuid4()
    event_id = uuid.uuid4()

    entity1 = Entity(
        id=entity1_id,
        name="John Doe",
        type="Person",
        aliases=["John", "Johnny"],
        confidence=0.9,
        evidence=base_evidence,
    )
    entity2 = Entity(
        id=entity2_id,
        name="ACME Corp",
        type="Organization",
        confidence=0.8,
        evidence=base_evidence,
    )

    relationship = Relationship(
        id=uuid.uuid4(),
        subject=entity1_id,
        predicate="works at",
        object=entity2_id,
        confidence=0.95,
        evidence=base_evidence,
    )

    event = Event(
        id=event_id,
        event_type="Acquisition",
        participants=[entity1_id, entity2_id],
        timestamp=datetime.now(),
        confidence=0.85,
        evidence=base_evidence,
    )

    memory = MemoryIR(
        entities=[entity1, entity2],
        relationships=[relationship],
        events=[event],
    )

    result = validator.validate(memory)
    assert result.valid is True
    assert len(result.issues) == 0

    report = format_report(result)
    assert report == "Validation Successful"


def test_duplicate_entity_ids(validator, base_evidence):
    dup_id = uuid.uuid4()
    entity1 = Entity(id=dup_id, name="Alice", type="Person", confidence=1.0, evidence=base_evidence)
    entity2 = Entity(id=dup_id, name="Bob", type="Person", confidence=1.0, evidence=base_evidence)

    memory = MemoryIR(entities=[entity1, entity2])
    result = validator.validate(memory)

    assert result.valid is False
    assert any(i.code == "ENTITY_DUPLICATE_ID" and i.location == "entities[1]" for i in result.issues)


def test_duplicate_relationship_ids(validator, base_evidence):
    e1 = Entity(name="Alice", confidence=1.0, evidence=base_evidence)
    e2 = Entity(name="Bob", confidence=1.0, evidence=base_evidence)

    dup_id = uuid.uuid4()
    r1 = Relationship(id=dup_id, subject=e1.id, predicate="knows", object=e2.id, confidence=0.8, evidence=base_evidence)
    r2 = Relationship(id=dup_id, subject=e2.id, predicate="likes", object=e1.id, confidence=0.7, evidence=base_evidence)

    memory = MemoryIR(entities=[e1, e2], relationships=[r1, r2])
    result = validator.validate(memory)

    assert result.valid is False
    assert any(i.code == "RELATIONSHIP_DUPLICATE_ID" and i.location == "relationships[1]" for i in result.issues)


def test_duplicate_event_ids(validator, base_evidence):
    e1 = Entity(name="Alice", confidence=1.0, evidence=base_evidence)
    dup_id = uuid.uuid4()
    ev1 = Event(id=dup_id, event_type="meeting", participants=[e1.id], confidence=0.9, evidence=base_evidence)
    ev2 = Event(id=dup_id, event_type="call", participants=[e1.id], confidence=0.9, evidence=base_evidence)

    memory = MemoryIR(entities=[e1], events=[ev1, ev2])
    result = validator.validate(memory)

    assert result.valid is False
    assert any(i.code == "EVENT_DUPLICATE_ID" and i.location == "events[1]" for i in result.issues)


def test_empty_entity_name(validator, base_evidence):
    e1 = Entity(name="", type="Person", confidence=1.0, evidence=base_evidence)
    e2 = Entity(name="   ", type="Person", confidence=1.0, evidence=base_evidence)

    memory = MemoryIR(entities=[e1, e2])
    result = validator.validate(memory)

    assert result.valid is False
    issues = [i for i in result.issues if i.code == "ENTITY_NAME_EMPTY"]
    assert len(issues) == 2
    assert issues[0].location == "entities[0]"
    assert issues[1].location == "entities[1]"


def test_empty_predicates(validator, base_evidence):
    e1 = Entity(name="Alice", confidence=1.0, evidence=base_evidence)
    e2 = Entity(name="Bob", confidence=1.0, evidence=base_evidence)

    r1 = Relationship(subject=e1.id, predicate="", object=e2.id, confidence=0.8, evidence=base_evidence)
    r2 = Relationship(subject=e1.id, predicate="   ", object=e2.id, confidence=0.8, evidence=base_evidence)

    memory = MemoryIR(entities=[e1, e2], relationships=[r1, r2])
    result = validator.validate(memory)

    assert result.valid is False
    issues = [i for i in result.issues if i.code == "RELATIONSHIP_EMPTY_PREDICATE"]
    assert len(issues) == 2
    assert issues[0].location == "relationships[0]"
    assert issues[1].location == "relationships[1]"


def test_invalid_relationship_references(validator, base_evidence):
    e1 = Entity(name="Alice", confidence=1.0, evidence=base_evidence)

    # Subject is valid, Object is some random UUID
    invalid_obj_id = uuid.uuid4()
    r = Relationship(subject=e1.id, predicate="knows", object=invalid_obj_id, confidence=0.8, evidence=base_evidence)

    memory = MemoryIR(entities=[e1], relationships=[r])
    result = validator.validate(memory)

    assert result.valid is False
    assert any(i.code == "RELATIONSHIP_INVALID_OBJECT" and i.location == "relationships[0]" for i in result.issues)


def test_invalid_event_participants(validator, base_evidence):
    e1 = Entity(name="Alice", confidence=1.0, evidence=base_evidence)
    invalid_part_id = uuid.uuid4()

    ev = Event(event_type="meeting", participants=[e1.id, invalid_part_id], confidence=0.9, evidence=base_evidence)

    memory = MemoryIR(entities=[e1], events=[ev])
    result = validator.validate(memory)

    assert result.valid is False
    assert any(i.code == "EVENT_INVALID_PARTICIPANT" and i.location == "events[0]" for i in result.issues)


def test_duplicate_aliases(validator, base_evidence):
    e = Entity(name="Alice", type="Person", aliases=["Ally", "Ally"], confidence=1.0, evidence=base_evidence)

    memory = MemoryIR(entities=[e])
    result = validator.validate(memory)

    assert result.valid is False
    assert any(i.code == "ENTITY_DUPLICATE_ALIAS" and i.location == "entities[0]" for i in result.issues)


def test_missing_evidence(validator):
    # Using model_construct to bypass pydantic validation for instantiation
    e = Entity.model_construct(
        id=uuid.uuid4(),
        name="Alice",
        type="Person",
        aliases=[],
        confidence=1.0,
        evidence=None
    )

    memory = MemoryIR(entities=[e])
    result = validator.validate(memory)

    assert result.valid is False
    assert any(i.code == "EVIDENCE_MISSING" and i.location == "entities[0]" for i in result.issues)


def test_empty_evidence_fields(validator):
    e = Entity(
        name="Alice",
        type="Person",
        confidence=1.0,
        evidence=Evidence(message_id="", text="")
    )

    memory = MemoryIR(entities=[e])
    result = validator.validate(memory)

    assert result.valid is False
    assert any(i.code == "EVIDENCE_EMPTY_MESSAGE_ID" and i.location == "entities[0]" for i in result.issues)
    assert any(i.code == "EVIDENCE_EMPTY_TEXT" and i.location == "entities[0]" for i in result.issues)


def test_warnings_only_valid_result(validator, base_evidence):
    # Missing type generates a warning, but should be valid
    e = Entity(
        name="Alice",
        type=None,
        confidence=1.0,
        evidence=base_evidence
    )

    memory = MemoryIR(entities=[e])
    result = validator.validate(memory)

    assert result.valid is True
    assert len(result.issues) == 1
    assert result.issues[0].level == IssueLevel.WARNING
    assert result.issues[0].code == "ENTITY_TYPE_MISSING"

    report = format_report(result)
    assert "Validation Successful" in report
    assert "Warnings" in report
    assert "• ENTITY_TYPE_MISSING" in report
