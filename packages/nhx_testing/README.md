<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Helix Testing

Testing utilities for NeMo Helix services.

## Overview

This package provides testing utilities for NeMo Helix services, including:

- `create_test_client` - A context manager for creating test clients with in-memory storage

## Usage

```python
from fastapi.testclient import TestClient
from nhx.common.entities.client import EntityClient
from nhx.testing import create_test_client

# Use create_test_client for API testing
with create_test_client(MyService, client_type=TestClient) as client:
    response = client.get("/api/endpoint")

# Use create_test_client for service-level tests with EntityClient
with create_test_client(client_type=EntityClient) as entity_client:
    await entity_client.create(my_entity)
```

`create_test_client` runs sync and async requests on the application's lifespan
event loop, including requests from async tests or separate `asyncio.run` calls.
This keeps service-owned async connections on one loop through shutdown. Async
request bodies are read on the caller's loop one chunk at a time, as the application
requests them. Early responses stop reading the upload; server timeouts and caller
cancellation cancel pending body reads.
