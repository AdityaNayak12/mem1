import pytest
from uuid import uuid4
from datetime import datetime

from app.schemas.memory_ir import (
    Entity,
    Relationship,
    Event,
    Evidence,
    MemoryIR,
)
from app.validation import StructuralValidator


# -------------------------
# Helper Functions
# -------------------------

def make_evidence():
    return Evidence(
        message_id="msg-1",
        text="I'm building Mem1 using FastAPI."
    )


def make_entity(name="FastAPI"):
    return Entity(
        id=uuid4(),
        name=name,
        type="Framework",
        confidence=0.95,
        evidence=make_evidence()
    )


# -------------------------
# Tests
# -------------------------

def test_valid_memory_ir():
    entity1 = make_entity("Mem1")
    entity2 = make_entity("FastAPI")

    relationship = Relationship(
        id=uuid4(),
        subject=entity1.id,
        predicate="USES",
        object=entity2.id,
        confidence=0.9,
        evidence=make_evidence()
    )

    event = Event(
        id=uuid4(),
        event_type="Development",
        participants=[entity1.id],
        timestamp=datetime.now(),
        confidence=0.95,
        evidence=make_evidence()
    )

    memory = MemoryIR(
        entities=[entity1, entity2],
        relationships=[relationship],
        events=[event]
    )

    result = StructuralValidator().validate(memory)

    assert result.valid
    assert len(result.issues) == 0


def test_empty_entity_name():
    entity = make_entity("")
    memory = MemoryIR(entities=[entity])

    result = StructuralValidator().validate(memory)

    assert not result.valid
    assert any(issue.code == "ENTITY_NAME_EMPTY" for issue in result.issues)


def test_duplicate_entity_ids():
    duplicate_id = uuid4()

    e1 = Entity(
        id=duplicate_id,
        name="A",
        type="Project",
        confidence=0.9,
        evidence=make_evidence()
    )

    e2 = Entity(
        id=duplicate_id,
        name="B",
        type="Framework",
        confidence=0.9,
        evidence=make_evidence()
    )

    memory = MemoryIR(entities=[e1, e2])

    result = StructuralValidator().validate(memory)

    assert not result.valid
    assert any(issue.code == "ENTITY_DUPLICATE_ID" for issue in result.issues)


def test_invalid_relationship_reference():
    entity = make_entity()

    relationship = Relationship(
        id=uuid4(),
        subject=entity.id,
        predicate="USES",
        object=uuid4(),  # does not exist
        confidence=0.9,
        evidence=make_evidence()
    )

    memory = MemoryIR(
        entities=[entity],
        relationships=[relationship]
    )

    result = StructuralValidator().validate(memory)

    assert not result.valid
    assert any(
        issue.code == "RELATIONSHIP_INVALID_OBJECT"
        for issue in result.issues
    )


def test_empty_predicate():
    e1 = make_entity("Mem1")
    e2 = make_entity("FastAPI")

    relationship = Relationship(
        id=uuid4(),
        subject=e1.id,
        predicate="",
        object=e2.id,
        confidence=0.9,
        evidence=make_evidence()
    )

    memory = MemoryIR(
        entities=[e1, e2],
        relationships=[relationship]
    )

    result = StructuralValidator().validate(memory)

    assert not result.valid
    assert any(
        issue.code == "RELATIONSHIP_EMPTY_PREDICATE"
        for issue in result.issues
    )


def test_invalid_event_participant():
    event = Event(
        id=uuid4(),
        event_type="Deployment",
        participants=[uuid4()],
        timestamp=datetime.now(),
        confidence=0.9,
        evidence=make_evidence()
    )

    memory = MemoryIR(events=[event])

    result = StructuralValidator().validate(memory)

    assert not result.valid
    assert any(
        issue.code == "EVENT_INVALID_PARTICIPANT"
        for issue in result.issues
    )


def test_empty_evidence_text():
    entity = Entity(
        id=uuid4(),
        name="FastAPI",
        type="Framework",
        confidence=0.9,
        evidence=Evidence(
            message_id="msg-1",
            text=""
        )
    )

    memory = MemoryIR(entities=[entity])

    result = StructuralValidator().validate(memory)

    assert not result.valid
    assert any(
        issue.code == "EVIDENCE_EMPTY_TEXT"
        for issue in result.issues
    )


def test_duplicate_aliases():
    entity = Entity(
        id=uuid4(),
        name="FastAPI",
        type="Framework",
        aliases=["FA", "FA"],
        confidence=0.9,
        evidence=make_evidence()
    )

    memory = MemoryIR(entities=[entity])

    result = StructuralValidator().validate(memory)

    assert not result.valid
    assert any(
        issue.code == "ENTITY_DUPLICATE_ALIAS"
        for issue in result.issues
    )