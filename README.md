# Mem1

Mem1 is a persistent memory subsystem for LLM agents. Unlike traditional RAG (Retrieval-Augmented Generation) systems that retrieve documents, Mem1 maintains a structured, evolving knowledge graph representing long-term memories extracted from conversations.

## Pipeline Architecture (v0.1)

```
Conversation ──> Extraction ──> Memory IR ──> Structural Validation ──> Commit Engine ──> Neo4j
```

- **Extraction**: SLM/LLM parses conversation transcripts to produce structured representations.
- **Memory IR**: Pydantic models containing `Entity`, `Relationship`, `Event` definitions, as well as `Evidence` (provenance mapping message IDs to text snippets).
- **Structural Validator**: A deterministic, offline validator verifying structural correctness, internal consistency (e.g., relationship and event participant mappings pointing to valid entities), alias duplicates, and evidence availability.
- **Commit Engine**: Idempotently commits validated memories into Neo4j using Cypher `MERGE` queries.

## Current State

The project structure has been fully scaffolded, and the **Structural Validation Subsystem** is complete and fully tested:
- **`app/validation/models.py`**: Structured Pydantic validation issue and result models.
- **`app/validation/rules.py`**: Clean, deterministic validation rules for entities, relationships, and events.
- **`app/validation/validator.py`**: Orchestration validator class.
- **`app/validation/report.py`**: Human-readable test and report formatter.
- **`tests/test_validation.py`**: 12 comprehensive unit tests verifying error and warning conditions.

## Project Structure

```
mem1/
├── app/
│   ├── api/                  # FastAPI routes (placeholder)
│   ├── core/                 # App configuration & logger (placeholder)
│   ├── extraction/           # Extraction logic (placeholder)
│   ├── graph/                # Neo4j operations & Cypher queries (placeholder)
│   ├── schemas/
│   │   └── memory_ir.py      # Core Pydantic schemas (Entities, Relationships, etc.)
│   ├── validation/           # Structural Validation Subsystem
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── report.py
│   │   ├── rules.py
│   │   └── validator.py
│   └── main.py               # Main API entrypoint
├── docker-compose.yml        # Neo4j Docker Compose file
├── pyproject.toml            # Package dependency & pytest configuration
└── tests/
    └── test_validation.py    # Unit tests for the validation rules
```

## Quick Start

### 1. Installation
Install project dependencies in a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Running Tests
Execute the structural validation unit tests:
```bash
pytest tests/test_validation.py
```
