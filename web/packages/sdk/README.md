<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# @nemo/sdk

A collection of generated types and hooks for NeMo Helix.

This package uses [Orval](https://orval.dev/) to generate TypeScript types and React Query hooks from OpenAPI specifications or interacting with various NeMo Helix.

## Features

- Type-safe API clients generated from OpenAPI specifications
- React Query hooks for data fetching and caching
- Support for multiple microservices:
  - Customizer
  - Deployment Management
  - Entity Store
  - Evaluation

## Prerequisites

This package is expected to be used in a Vite app.

## Installation

```bash
pnpm add @nemo/sdk
```

## Environment Setup

The generated hooks require the following environment variable in your Vite app:

```env
VITE_PLATFORM_BASE_URL=<your-platform-url>
```

## Usage

### Generating Types and Hooks

You can generate types and hooks for all microservices using:

```bash
pnpm gen:all
```

Or generate for specific microservices:

```bash
# Generate for Customizer
pnpm gen:customizer

# Generate for Deployment Management
pnpm gen:deployment-management

# Generate for Entity Store
pnpm gen:entity-store

# Generate for Evaluation
pnpm gen:evaluation
```

### Using Generated Hooks

The generated hooks can be used in your React components:

```typescript
import { useListModels } from '@nemo/sdk/entity-store/generated/api';

function MyComponent() {
  const { data, isLoading } = useListModels();
  // ... use the data
}
```
