from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    message_id: str
    text: str
    start: int | None = None
    end: int | None = None


class Entity(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    type: str | None = None
    aliases: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: Evidence


class Relationship(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    subject: UUID
    predicate: str
    object: UUID
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: Evidence


class Event(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    event_type: str
    participants: list[UUID]
    timestamp: datetime | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: Evidence


class MemoryIR(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)