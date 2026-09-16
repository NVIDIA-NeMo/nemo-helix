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

# Use create_test_client with SDK access
with create_test_client(MyService) as sdk:
    sdk.entities.create(...)

# Use create_test_client for service-level tests with EntityClient
with create_test_client(client_type=EntityClient) as entity_client:
    await entity_client.create(my_entity)
```
