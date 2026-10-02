<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Entity Store

Generic entity storage service with schema-agnostic design, part of the NeMo Helix Core infrastructure.

## Table of Contents

- [Entity Store](#entity-store)
  - [Overview](#overview)
  - [Architecture](#architecture)
  - [Endpoints](#endpoints)
    - [Health](#health)
    - [Workspace Endpoints (v2)](#workspace-endpoints-v2)
    - [Project Endpoints (v2)](#project-endpoints-v2)
    - [Entity Endpoints (v2)](#entity-endpoints-v2)
  - [Entity Format](#entity-format)
  - [Search and Filtering](#search-and-filtering)
    - [Comparison Operators](#comparison-operators)
    - [Logical Operators](#logical-operators)
    - [Search Examples](#search-examples)
  - [Testing](#testing)

## Overview

Entity Store provides a schema-agnostic storage system for all entity types in the NeMo Helix. It treats entity-specific data as opaque JSONB, enabling type-safe client SDKs to be built on top while keeping the storage layer completely generic.

### Key Features

- **Schema-Agnostic**: Single `entities` table stores all entity types using JSONB
- **Type-Safe SDKs**: Client libraries provide compile-time type safety via Pydantic
- **Workspace-Based Organization**: Entities organized by workspace with name-based identity
- **Project Support**: Entities can be grouped into projects within a workspace
- **Nested Entities**: Parent-child relationships via `parent` field
- **Cross-Workspace Queries**: List entities across all accessible workspaces
- **Advanced Search**: Rich filtering with comparison and logical operators
- **Optimized Performance**: Specialized indexes for efficient querying and JSONB filtering

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  API Layer (FastAPI)                                            │
│  /v2/workspaces/{workspace}/entities/{entity_type}/{name}       │
│  /v2/workspaces/{workspace}/projects/{name}                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Depends() injection
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Repository Interfaces (ABC)                                    │
│  src/entities/app/repository/                                   │
│  - WorkspaceRepositoryInterface                                 │
│  - EntityRepositoryInterface                                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ implements
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  SQLAlchemy Implementation                                      │
│  src/entities/app/repository/sqlalchemy/                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ models.py                                                  │ │
│  │ - DBWorkspace   (workspaces table)                         │ │
│  │ - DBEntity      (entities table + JSONB)                   │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │ PostgreSQL + asyncpg
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  PostgreSQL Database                                            │
│  - workspaces (id, description, created_at, updated_at)         │
│  - entities (workspace, entity_type, name, parent, project,     │
│              data, created_at, updated_at)                      │
│      → FK workspace                                             │
│      → FK project (optional)                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Endpoints

### Health

- `GET /health` - Health check endpoint

### Workspace Endpoints (v2)

- `POST /v2/workspaces` - Create workspace
- `GET /v2/workspaces` - List all workspaces (filtered by access)
- `GET /v2/workspaces/{name}` - Get workspace by name
- `PUT /v2/workspaces/{name}` - Update workspace
- `DELETE /v2/workspaces/{name}` - Delete workspace

### Project Endpoints (v2)

Projects provide organizational grouping for entities within a workspace.

- `POST /v2/workspaces/{workspace}/projects` - Create project
- `GET /v2/workspaces/{workspace}/projects` - List projects in workspace
- `GET /v2/workspaces/{workspace}/projects/{name}` - Get project by name
- `PUT /v2/workspaces/{workspace}/projects/{name}` - Update project
- `DELETE /v2/workspaces/{workspace}/projects/{name}` - Delete project

### Entity Endpoints (v2)

Entities use name-based identity within workspace and entity type scope.

**Name-Based Operations (Primary):**

- `POST /v2/workspaces/{workspace}/entities/{entity_type}` - Create entity
- `GET /v2/workspaces/{workspace}/entities/{entity_type}` - List entities
- `GET /v2/workspaces/{workspace}/entities/{entity_type}/{name}` - Get entity by name
- `PUT /v2/workspaces/{workspace}/entities/{entity_type}/{name}` - Update entity
- `DELETE /v2/workspaces/{workspace}/entities/{entity_type}/{name}` - Delete entity

**Cross-Workspace Queries:**

- `GET /v2/workspaces/-/entities/{entity_type}` - List entities across all accessible workspaces

**ID-Based Operations:**

- `GET /v2/entities/{id}` - Get entity by ID

## Entity Format

All entities follow this standardized format:

```json
{
  "id": "customization_config-5Q2LoF8z8M9JZxZsHwJKNn",
  "workspace": "ml-team",
  "entity_type": "customization_config",
  "name": "my-config",
  "parent": null,
  "project": "llm-training",
  "data": {
    "target_id": "llama-2-7b",
    "training_options": {"learning_rate": 0.01}
  },
  "created_at": "2025-09-24T16:45:00Z",
  "updated_at": "2025-09-24T16:45:00Z"
}
```

| Field | Description |
|-------|-------------|
| `id` | System-generated unique identifier |
| `workspace` | Workspace the entity belongs to |
| `entity_type` | Type identifier (e.g., `customization_config`, `model`) |
| `name` | User-provided name, unique within (workspace, entity_type, parent) |
| `parent` | Parent entity ID for nested entities (optional) |
| `project` | Project name for organizational grouping (optional) |
| `data` | Entity-specific payload as JSONB |
| `created_at` | Creation timestamp |
| `updated_at` | Last modification timestamp |

## Search and Filtering

The Entity Store supports advanced filtering via the `search` query parameter. Filters can be provided as JSON or bracket notation.

### Comparison Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equal (default) | `{"name": "llama"}` or `{"name": {"$eq": "llama"}}` |
| `$like` | Contains/pattern match | `{"name": {"$like": "llama"}}` |
| `$lt` | Less than | `{"created_at": {"$lt": "2025-01-01"}}` |
| `$lte` | Less than or equal | `{"created_at": {"$lte": "2025-01-01"}}` |
| `$gt` | Greater than | `{"created_at": {"$gt": "2025-01-01"}}` |
| `$gte` | Greater than or equal | `{"created_at": {"$gte": "2025-01-01"}}` |
| `$in` | In list | `{"name": {"$in": ["llama", "mistral"]}}` |
| `$nin` | Not in list | `{"name": {"$nin": ["llama", "mistral"]}}` |

### Logical Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$and` | All conditions must match | `{"$and": [{"name": "llama"}, {"status": "active"}]}` |
| `$or` | Any condition must match | `{"$or": [{"name": "llama"}, {"name": "mistral"}]}` |
| `$not` | Negate condition | `{"$not": {"name": "llama"}}` |

### Search Examples

**JSON notation:**

```bash
# Filter by name
GET /v2/workspaces/default/entities/model?search={"name":"llama"}

# Filter with operator
GET /v2/workspaces/default/entities/model?search={"name":{"$like":"llama"}}

# Multiple conditions (AND)
GET /v2/workspaces/default/entities/model?search={"name":{"$like":"llama"},"created_at":{"$gte":"2025-01-01"}}

# OR condition
GET /v2/workspaces/default/entities/model?search={"$or":[{"name":"llama"},{"name":"mistral"}]}

# Filter on nested data fields
GET /v2/workspaces/default/entities/customization_config?search={"data.target_id":"llama-2-7b"}
```

**Bracket notation:**

```bash
# Simple filter
GET /v2/workspaces/default/entities/model?search[name]=llama

# With operator
GET /v2/workspaces/default/entities/model?search[name][$like]=llama

# Multiple filters
GET /v2/workspaces/default/entities/model?search[name][$like]=llama&search[created_at][$gte]=2025-01-01

# IN operator (comma-separated)
GET /v2/workspaces/default/entities/model?search[name][$in]=llama,mistral
```

## Testing

Run all tests from the service directory:

```bash
uv run --frozen pytest tests/ -v
```

Run specific test files:

```bash
# Unit tests
uv run --frozen pytest tests/test_search.py -v

# Integration tests
uv run --frozen pytest tests/integration/ -v
```