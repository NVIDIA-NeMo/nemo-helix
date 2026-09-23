<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Testing Guide for NeMo Helix

This document describes the testing strategy, structure, and best practices for the NeMo Helix repository.

## Table of Contents

- [Test Categories](#test-categories)
- [Choosing Between Integration and E2E Tests](#choosing-between-integration-and-e2e-tests)
- [Test Organization](#test-organization)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Test Markers](#test-markers)
- [Coverage Reporting](#coverage-reporting)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Test Categories

We organize tests into six categories based on their scope and purpose:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TEST PYRAMID                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Scope          Test Type                  Infrastructure    Speed          │
│  ─────          ─────────                  ──────────────    ─────          │
│                                                                             │
│                 ┌───────────────┐                                           │
│  Deployed       │  E2E Tests    │          Helm/Docker      Slowest         │
│  Infra          │     (e2e/)    │          + GPUs           (hours)         │
│                 └───────┬───────┘                                           │
│                         │                                                   │
│                 ┌───────▼───────┐                                           │
│  Service        │  Integration  │          ASGI + DB        Medium          │
│                 │    Tests      │          (mock others)    (seconds)       │
│                 │  (svc/tests/) │                                           │
│                 └───────┬───────┘                                           │
│                         │                                                   │
│                 ┌───────▼───────┐                                           │
│  Isolated       │  Unit Tests   │          None             Fast            │
│                 │  (default)    │          (all mocked)     (ms)            │
│                 └───────────────┘                                           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Other categories: Regression, Infrastructure, Canary                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. Unit Tests (Default)

**Objective**: Test single classes or functions quickly without any infrastructure dependencies.

**Characteristics**:

- Fast execution (milliseconds to seconds)
- No external dependencies (databases, networks, file systems)
- Use mocks and stubs for dependencies
- Test one thing at a time
- Do not require markers

**Example**:

```python
def test_calculate_sum():
    """Unit test for a simple calculation function."""
    result = calculate_sum(2, 3)
    assert result == 5
```

**Location**: Adjacent to the code being tested in `tests/` directories within packages and services.

### 2. Integration Tests

**Objective**: Test service interfaces and interactions with their direct dependencies.

**Characteristics**:

- Test how a service's components work together
- Uses ASGI test client for fast, in-process testing
- Mocks external services via SDK calls
- May use real databases for the service under test
- Slower than unit tests (seconds to minutes)
- Require `@pytest.mark.integration` marker
- Always run inside the respective service's test directory

**Examples**:

- Testing CRUD endpoints are working properly
- Testing file upload and retrieval
- Testing OpenAPI spec compliance
- Testing service-specific business logic
- Testing cross-service interactions (with mocked external services)

**Example**:

```python
@pytest.mark.integration
def test_create_and_fetch_entity(client, db_session):
    """Integration test for entity CRUD operations."""
    # Create entity
    response = client.post("/v1/entities", json={"name": "test"})
    assert response.status_code == 201
    entity_id = response.json()["id"]

    # Fetch entity
    response = client.get(f"/v1/entities/{entity_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "test"
```

**Location**: `services/<service>/tests/integration/` or `packages/<package>/tests/integration/`

### 3. End-to-End (E2E) Tests

**Objective**: Ensure that services work together correctly when running as a real platform process.

**Characteristics**:

- Start the platform via `nemo services run` (real process, real ports)
- Hit services with an external HTTP client (the NeMoHelix SDK)
- Test startup machinery, port binding, config resolution, and cross-service workflows
- Slower than integration tests (tens of seconds for startup) but faster than Docker/K8s e2e

**How to run**:

```bash
# Start services, run tests, stop services (all automatic)
make test-e2e

# Or manually
uv run --frozen pytest e2e -v --run-e2e

# If you already have services running
NHX_BASE_URL=http://localhost:8080 uv run --frozen pytest e2e -v --run-e2e
```

**Prerequisites**: `make bootstrap` must have been run. The harness spawns `nemo services run`
on a free port, so it won't conflict with your dev instance.

**When to Write E2E Tests**:

Write E2E tests when you need to:
- **Verify cross-service workflows**: Test operations that span multiple services (e.g., create workspace → upload file → run job → get results)
- **Validate the real startup path**: Ensure config resolution, service discovery, and health checks work
- **Test authentication/authorization**: Verify role-based access control across multiple services
- **Ensure service integration**: Verify that services work together correctly end-to-end

**What NOT to E2E Test**:
- Single service APIs (use integration tests)
- Business logic within a service (use unit/integration tests)
- Every permutation of inputs (E2E should focus on critical paths)
- Implementation details (test user-visible behavior)

**Location**: `e2e/` (root-level)

### 4. Infrastructure Tests

**Objective**: Ensure services are compatible with a variety of customer infrastructure.

**Characteristics**:

- Test deployment scenarios
- Test infrastructure integrations
- May require specific cloud providers or configurations
- Require `@pytest.mark.infrastructure` marker

**Examples**:

- Testing deployment with KAI k8s scheduler
- Testing secrets service backed by AWS KMS
- Testing different storage backends

**Example**:

```python
@pytest.mark.infrastructure
def test_deploy_with_kai_scheduler():
    """Test deployment using KAI kubernetes scheduler."""
    # Test infrastructure compatibility
    pass
```

**Location**: `tests/infrastructure/` (to be created as needed)

### 5. Regression Tests

**Objective**: Ensure baseline product and feature functionality is maintained.

**Characteristics**:

- Test specific features or microservices
- Ensure functionality doesn't regress over time
- May be slower than unit tests
- Require `@pytest.mark.regression` marker

**Examples**:

- SDG service generates quality data meeting criteria
- Evaluator calculations are correct
- Fine-tuned models reach target accuracy

**Example**:

```python
@pytest.mark.regression
def test_sdg_quality_threshold():
    """Regression test to ensure SDG maintains quality standards."""
    data = generate_synthetic_data(config)
    quality_score = evaluate_quality(data)
    assert quality_score >= 0.85, "SDG quality dropped below threshold"
```

**Location**: `tests/regression/` or service-specific test directories

### 6. Canary Tests

**Objective**: Test deployed integration environments such as top-of-tree.

**Characteristics**:

- Run against continuously deployed environments
- Monitor environment stability
- Check performance baselines
- Require `@pytest.mark.canary` marker

**Examples**:

- Health check of staging environment
- Performance monitoring of production services
- Smoke tests on ToT deployment

**Example**:

```python
@pytest.mark.canary
def test_staging_environment_health():
    """Canary test to verify staging environment is healthy."""
    response = requests.get(f"{STAGING_URL}/health")
    assert response.status_code == 200
```

**Location**: `tests/canary/` (to be created as needed)

## Choosing Between Integration and E2E Tests

To understand the difference between Integration and End-to-End (E2E) testing, it helps to think about them in terms of scope and perspective. While both ensure that different parts of your system work together, they do so at different levels of the Testing Pyramid.

### Integration Testing: The "Plumbing"

Integration testing focuses on the interfaces between components. It asks: **"Does the data flow correctly from Point A to Point B?"**

- **Scope**: Usually two or more modules within a service (e.g., an API and its database)
- **Perspective**: Developer-centric - checks if the "contracts" between components are honored
- **Environment**: Often uses mocks or stubs for external services to keep tests fast and isolated
- **Speed**: Fast (seconds)
- **Example**: Testing that when your Files Service receives an upload request, the file metadata is correctly stored in the database

### End-to-End (E2E) Testing: The "User Journey"

E2E testing validates the entire system from start to finish. It asks: **"Can the user actually achieve their goal?"**

- **Scope**: The complete stack—multiple services, databases, and real infrastructure
- **Perspective**: User-centric - simulates real workflows
- **Environment**: Production-like environment with real services (Docker/K8s) - no mocks
- **Speed**: Slow (minutes)
- **Example**: A test that creates a workspace, uploads data, runs a training job, monitors progress, and verifies the model output

### When to Use Each Type

**Write Integration Tests when:**
- Testing API contracts and response shapes
- Validating business logic within a service
- Testing database operations (CRUD, queries, transactions)
- You can mock external dependencies
- You need fast feedback during development

**Write E2E Tests when:**
- Testing complete user workflows across services
- Validating real infrastructure (jobs, storage, inference)
- Testing authentication/authorization across services
- Verifying async operations and event propagation
- Ensuring services integrate correctly in production

### Quick Comparison

| Aspect | Integration Test | E2E Test |
|--------|-----------------|----------|
| **Question** | "Does this service work correctly?" | "Can the user complete their task?" |
| **Scope** | Single service | Full system |
| **Speed** | Seconds | Minutes |
| **Dependencies** | Mocked | Real |
| **When to Run** | Every commit | Nightly/pre-release |

### Example: Testing File Upload

**Integration Test** (focuses on the Files Service contract):
```python
@pytest.mark.integration
def test_file_upload_api(client):
    """Test that file upload API correctly stores metadata."""
    response = client.post("/v2/files", files={"file": test_data})
    assert response.status_code == 201
    assert response.json()["filename"] == "test.txt"
    # Verifies API contract, not cross-service behavior
```

**E2E Test** (focuses on complete user workflow):
```python
def test_data_pipeline(sdk: NeMoHelix):
    """Test complete data pipeline from upload to results."""
    # User uploads training data
    file = sdk.files.upload(workspace="default", file=training_data)

    # User creates and runs training job
    job = sdk.jobs.create(
        workspace="default",
        spec={"input_file": file.id, "model": "gpt"}
    )

    # Wait for job completion
    wait_for_completion(sdk, job.id)

    # User retrieves trained model
    model = sdk.models.retrieve(job.spec.output.name)
    assert model.status == "ready"
    # Tests the complete user journey
```



## Test Organization

### Directory Structure

```
nhx/
├── pytest.ini                    # Root pytest configuration
├── conftest.py                   # Shared fixtures for all tests
├── e2e/                          # E2E tests (testcontainers-based)
│   ├── configs/                  # NeMo Helix configuration files
│   ├── conftest.py               # E2E fixtures (backend, sdk, workspace)
│   └── test_*.py                 # E2E test files
├── tests/                        # Root-level tests
│   └── ...
├── packages/                     # Library packages
│   ├── nhx_platform/             # Legacy platform task entrypoint package
│   │   └── tests/
│   ├── nhx_common/
│   │   └── tests/                # Unit tests (primarily)
│   ├── data_designer/
│   │   └── tests/
│   │       ├── unit/             # Optional: explicit unit test directory
│   │       ├── integration/      # Package-level integration tests
│   │       └── conftest.py       # Package-specific fixtures
│   └── ...
└── services/                     # Microservices
    ├── evaluator/
    │   └── tests/
    │       ├── unit/              # Service unit tests
    │       ├── integration/       # Integration tests
    │       └── conftest.py        # Service-specific fixtures
    └── ...
```

### Configuration Files

- **pytest.ini**: Root configuration with markers and global settings
- **conftest.py**: Shared fixtures and test hooks
- **pyproject.toml**: Coverage configuration and Python package metadata
- **Makefile**: Convenient test execution targets

## Running Tests

### Quick Start

Run all unit tests (fastest):
```bash
make test
# or
make test-unit
# or directly with pytest (discovers all packages and services)
uv run pytest
```

For a detailed summary of what's being tested:
```bash
# Show test discovery summary
uv run python tools/run_all_tests.py --summary-only

# Run all unit tests with detailed output
uv run python tools/run_all_tests.py
```

### Run Specific Test Categories

```bash
# Unit tests only (default - no marker needed)
make test-unit
uv run pytest -v

# Unit tests with summary
uv run python tools/run_all_tests.py

# Integration tests
make test-integration
uv run pytest -v -m integration

# End-to-end tests (starts nemo services automatically)
make test-e2e
uv run --frozen pytest e2e -v --run-e2e

# E2E against an already-running instance
NHX_BASE_URL=http://localhost:8080 uv run --frozen pytest e2e -v --run-e2e

# Regression tests
make test-regression
uv run pytest -v -m regression

# All tests except canary and CI-skipped
make test-all
uv run pytest -v -m "not canary and not skip_in_ci"

# Canary tests (for deployed environments)
make test-canary
uv run pytest -v -m canary
```

### Run Tests for Specific Components

```bash
# Test a specific package
make test-package PACKAGE=nhx_common
uv run pytest -v packages/nhx_common/tests/

# Test a specific service
make test-service SERVICE=guardrails

# Test a specific file
uv run pytest -v path/to/test_file.py

# Test a specific function
uv run pytest -v path/to/test_file.py::test_function_name
```

### Advanced Test Options

```bash
# Run tests in debug mode (verbose, show output, show locals)
make test-debug
uv run pytest -vvv -s --tb=long --showlocals

# Run only failed tests from last run
make test-failed
uv run pytest -v --lf

# Run tests in parallel (faster, requires pytest-xdist)
uv run pytest -v -n auto

# Run fast tests only (exclude slow, e2e, integration)
make test-fast
uv run pytest -v -m "not slow and not e2e and not integration"

# List all tests without running them
make test-list
uv run pytest --collect-only -q

# Show available markers
make test-markers
uv run pytest --markers
```

## Coverage Reporting

### Generate Coverage Reports

```bash
# Run tests with coverage
make test-coverage

# View coverage report in browser
make test-coverage-report

# Generate specific format
uv run pytest --cov --cov-report=html  # HTML report
uv run pytest --cov --cov-report=term  # Terminal report
uv run pytest --cov --cov-report=xml   # XML report (for CI)
```

### Coverage Configuration

Coverage settings are in `pyproject.toml`:
- Source directories: `packages/`, `services/`, `src/`
- Omits test files, `__pycache__`, virtual environments
- Branch coverage enabled
- HTML report in `htmlcov/`
- XML report in `coverage.xml`

### Coverage Goals

- Start with baseline measurement
- Gradually increase coverage requirements
- Focus on critical paths first
- Aim for >80% coverage on business logic
- 100% coverage on critical security/financial code

## Writing Tests

### Test Naming Conventions

- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`

### Using Markers

```python
import pytest

# Unit test (no marker needed)
def test_simple_function():
    assert add(1, 2) == 3

# Integration test
@pytest.mark.integration
def test_database_operations(db_session):
    # Test database operations
    pass

# E2E test with timeout (deployed infrastructure)
@pytest.mark.e2e
@pytest.mark.timeout(1800)
def test_complete_workflow():
    # Test complete workflow
    pass

# Regression test
@pytest.mark.regression
def test_feature_baseline():
    # Test baseline functionality
    pass

# Slow test
@pytest.mark.slow
def test_large_dataset_processing():
    # Test that takes a long time
    pass

# Skip in CI
@pytest.mark.skip_in_ci
def test_local_only_feature():
    # Test that only runs locally
    pass

# Multiple markers
@pytest.mark.integration
@pytest.mark.slow
def test_slow_integration():
    pass
```

### Using Fixtures

```python
# Use shared fixtures from root conftest.py
def test_with_temp_dir(temp_dir):
    """Use temporary directory fixture."""
    file_path = temp_dir / "test.txt"
    file_path.write_text("test content")
    assert file_path.read_text() == "test content"

# Use service-specific fixtures
@pytest.mark.integration
def test_api_endpoint(client, db_session):
    """Use client and database fixtures."""
    response = client.post("/api/resource", json={"name": "test"})
    assert response.status_code == 201
```

### Async Tests

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await async_operation()
    assert result == expected_value
```

## Test Markers

Available markers are defined in `pytest.ini`:

- `integration`: Integration tests (tests service interfaces using ASGI transport)
- `e2e`: End-to-end tests (deployed infrastructure)
- `regression`: Regression tests
- `infrastructure`: Infrastructure tests
- `canary`: Canary tests for deployed environments
- `slow`: Tests that take a long time
- `skip_in_ci`: Tests that should be skipped in CI

## Best Practices

### General

1. **Test Pyramid**: Most tests should be unit tests, fewer integration tests, fewest e2e tests
2. **Fast Tests**: Keep tests fast - use mocks for external dependencies
3. **Independent Tests**: Each test should be independent and repeatable
4. **Clear Names**: Test names should clearly describe what they test
5. **One Assertion Per Concept**: Focus each test on one behavior
6. **Arrange-Act-Assert**: Structure tests clearly (setup, execute, verify)

### Unit Tests

- No external dependencies (network, database, filesystem)
- Use mocks and stubs for dependencies
- Test edge cases and error conditions
- Keep tests fast (<100ms ideally)
- High coverage (>80% for business logic)

### Integration Tests

- Test real interactions between components
- Use test databases/fixtures, not production data
- Clean up after tests (use fixtures with teardown)
- Be mindful of test duration
- Test error paths and edge cases at boundaries

### E2E Tests

- Test realistic user scenarios
- Use test environments, not production
- Make tests resilient to timing issues (use retries, waits)
- Clean up resources after tests
- Document environment requirements
- Use appropriate timeouts

### Fixtures

- Keep fixtures focused and reusable
- Use appropriate scope (function, class, module, session)
- Clean up resources in fixtures
- Document fixture behavior
- Place fixtures in conftest.py at appropriate level

### Mocking

- Mock at the boundary, not internal details
- Use `unittest.mock` or `pytest-mock`
- Verify mock calls when behavior matters
- Don't over-mock - integration tests should use real components

### Mocking Inference Calls

For tests that make inference calls through the Inference Gateway, use mock provider mode to return controlled responses without real LLM backends:

```python
from nhx.testing import ClientContext, add_mock_provider, create_test_client
from nhx.core.inference_gateway.service import InferenceGatewayService

@pytest.fixture
def mock_provider_clients() -> Generator[ClientContext, None, None]:
    with create_test_client(InferenceGatewayService, client_type=ClientContext) as clients:
        yield clients

def test_with_mock_llm(mock_provider_clients: ClientContext):
    provider = add_mock_provider(
        mock_provider_clients.sdk,
        workspace="default",
        name="my-llm",
        mock_response_body={"choices": [{"message": {"content": "Hello!"}}]},
    )
    response = mock_provider_clients.sdk.inference.gateway.provider.post(
        "v1/chat/completions",
        name=provider.name,
        workspace="default",
        body={"model": "test", "messages": []},
    )
    assert response["choices"][0]["message"]["content"] == "Hello!"
```

See [Mock Provider README](services/core/inference-gateway/src/nhx/core/inference_gateway/api/mock_provider/README.md) for complete documentation.

## Troubleshooting

### Common Issues

**Tests not found:**
- Check file naming convention (`test_*.py`)
- Check function naming convention (`test_*`)
- Verify test paths in `pytest.ini`

**Import errors:**
- Ensure virtual environment is activated
- Run `uv sync` to install dependencies
- Check `PYTHONPATH` if needed

**Slow tests:**
- Use `-v` to see which tests are slow
- Use `--durations=10` to show slowest tests
- Consider marking slow tests with `@pytest.mark.slow`

**Flaky tests:**
- Look for timing issues (use proper waits, not sleep)
- Check for test interdependencies
- Verify cleanup is happening
- Use `--lf` to re-run failed tests

**Coverage not working:**
- Ensure `pytest-cov` is installed
- Check source paths in `pyproject.toml`
- Run with `--cov` flag

### Getting Help

- Check pytest documentation: https://docs.pytest.org/
- Review test examples in the repository
- Ask in team channels

## CI/CD Integration

Tests are automatically run in CI/CD pipelines:

- **Pre-commit**: Fast unit tests
- **PR validation**: Unit + integration tests
- **Nightly**: All tests including e2e
- **Canary**: Continuous monitoring of deployed environments

## Migration from Old Test Structure

If you're updating existing tests:

1. Add appropriate markers (`@pytest.mark.integration`, etc.)
2. Update conftest.py to use shared fixtures where possible
3. Remove duplicate fixture definitions
4. Update test paths if needed
5. Run tests to verify they still work

## Future Enhancements

- [ ] Performance benchmarking tests
- [ ] Security-focused test suite
- [ ] Chaos engineering tests
- [ ] Load and stress tests
- [ ] Contract testing between services
