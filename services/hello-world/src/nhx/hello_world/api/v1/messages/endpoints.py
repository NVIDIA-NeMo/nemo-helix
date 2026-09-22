# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CRUD endpoints for HelloWorld message entities."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError
from nhx.common.service.dependencies import get_entity_client
from nhx.hello_world.api.v1.messages.schemas import CreateHelloWorldMessageRequest, UpdateHelloWorldMessageRequest
from nhx.hello_world.entities import HelloWorldMessage

router = APIRouter()
API_TAG = "Messages"


@router.post("/messages", response_model=HelloWorldMessage, status_code=201, tags=[API_TAG])
async def create_message(
    workspace: str,
    message_request: CreateHelloWorldMessageRequest,
    entity_store: EntityClient = Depends(get_entity_client),
) -> HelloWorldMessage:
    """Create a new HelloWorld message."""
    msg = HelloWorldMessage(
        name=message_request.name,
        workspace=workspace,
        description=message_request.description,
        message=message_request.message,
    )

    try:
        return await entity_store.create(msg)
    except EntityConflictError as e:
        raise HTTPException(
            status_code=409, detail=f"Message '{msg.name}' already exists in workspace '{workspace}'"
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/messages", response_model=List[HelloWorldMessage], tags=[API_TAG])
async def list_messages(
    workspace: str,
    entity_store: EntityClient = Depends(get_entity_client),
) -> List[HelloWorldMessage]:
    """List all HelloWorld messages in the workspace."""
    try:
        response = await entity_store.list(HelloWorldMessage, workspace=workspace)
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/messages/{name}", response_model=HelloWorldMessage, tags=[API_TAG])
async def get_message(
    workspace: str,
    name: str,
    entity_store: EntityClient = Depends(get_entity_client),
) -> HelloWorldMessage:
    """Get a HelloWorld message by name."""
    try:
        return await entity_store.get(HelloWorldMessage, name=name, workspace=workspace)
    except Exception as e:
        if "not found" in str(e).lower() or "404" in str(e):
            raise HTTPException(
                status_code=404,
                detail=f"Message '{name}' not found in workspace '{workspace}'",
            ) from e
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.patch("/messages/{name}", response_model=HelloWorldMessage, tags=[API_TAG])
async def update_message(
    workspace: str,
    name: str,
    update_data: UpdateHelloWorldMessageRequest,
    entity_store: EntityClient = Depends(get_entity_client),
) -> HelloWorldMessage:
    """Update a HelloWorld message."""
    # Get existing message
    try:
        existing = await entity_store.get(HelloWorldMessage, name=name, workspace=workspace)
    except Exception as e:
        if "not found" in str(e).lower() or "404" in str(e):
            raise HTTPException(
                status_code=404,
                detail=f"Message '{name}' not found in workspace '{workspace}'",
            ) from e
        raise HTTPException(status_code=500, detail=str(e)) from e

    for key, value in update_data.model_dump(exclude_unset=True).items():
        if value is not None:
            setattr(existing, key, value)

    # Apply updates to existing message
    try:
        return await entity_store.update(existing)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/messages/{name}", status_code=204, tags=[API_TAG])
async def delete_message(
    workspace: str,
    name: str,
    entity_store: EntityClient = Depends(get_entity_client),
) -> None:
    """Delete a HelloWorld message."""
    try:
        await entity_store.delete(HelloWorldMessage, name, workspace=workspace)
    except EntityConflictError as e:
        raise HTTPException(status_code=409, detail="Concurrent modification - please retry.") from e
    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Message '{name}' not found in workspace '{workspace}'",
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
