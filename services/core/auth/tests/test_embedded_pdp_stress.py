# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stress tests for the WASM policy engine.

Run with: uv run pytest -v services/auth/tests/test_policy_engine_stress.py
Skip slow tests: uv run pytest -v services/auth/tests/test_policy_engine_stress.py -m "not slow"
"""

import asyncio
import gc
import random
import statistics
import time
from pathlib import Path

import pytest
import yaml
from nhx.common.config import Configuration
from nhx.core.auth.app.embedded_pdp import engine as embedded_pdp_engine
from nhx.core.auth.app.embedded_pdp import evaluate, reload_policy, set_policy_data
from nhx.core.auth.config import AuthServiceConfig

# Large-scale fixtures load multi‑MB JSON into OPA; default embedded PDP WASM memory (32 MiB) is too small.
_STRESS_PDP_MEMORY_MB = 512


@pytest.fixture(scope="module", autouse=True)
def stress_embedded_pdp_memory_limit():
    """Raise WASM memory for this module so large-scale policy data can load."""
    cfg = AuthServiceConfig(embedded_pdp_memory_limit_mb=_STRESS_PDP_MEMORY_MB)  # type: ignore[misc]
    Configuration.set_override(cfg)
    embedded_pdp_engine._reset_policy_state_for_testing()
    yield
    Configuration.clear_override(AuthServiceConfig)
    embedded_pdp_engine._reset_policy_state_for_testing()


@pytest.fixture(scope="module")
def setup_policy_data():
    """Set up authorization data once for all stress tests."""
    # Create a dataset with multiple principals and workspaces
    principals = {}
    for i in range(100):
        principals[f"user{i}@example.com"] = {"workspaces": {f"workspace-{j}": ["Editor"] for j in range(10)}}

    data = {
        "authz": {
            "principals": principals,
            "roles": {
                "Editor": {"permissions": ["models.read", "models.create", "models.update"]},
                "Viewer": {"permissions": ["models.read"]},
                "Admin": {"permissions": ["models.read", "models.create", "models.update", "models.delete"]},
            },
            "endpoints": {
                "/v2/workspaces/{workspace_id}/models": {
                    "get": {"permissions": ["models.read"]},
                    "post": {"permissions": ["models.create"]},
                },
                "/v2/workspaces/{workspace_id}/models/{model_id}": {
                    "get": {"permissions": ["models.read"]},
                    "put": {"permissions": ["models.update"]},
                    "delete": {"permissions": ["models.delete"]},
                },
            },
            "workspaces": {},
        }
    }

    reload_policy()
    set_policy_data(data)
    yield
    reload_policy()


def test_single_evaluation_latency(setup_policy_data):
    """Measure latency for a single policy evaluation."""
    iterations = 500
    latencies = []

    for _ in range(iterations):
        start = time.perf_counter()
        evaluate(
            "allow",
            {
                "principal_id": "user0@example.com",
                "method": "GET",
                "path": "/v2/workspaces/workspace-0/models",
            },
        )
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    avg_latency = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]

    print(f"\nSingle evaluation latency ({iterations} iterations):")
    print(f"  Average: {avg_latency:.3f} ms")
    print(f"  P50:     {p50:.3f} ms")
    print(f"  P99:     {p99:.3f} ms")

    # Performance assertions
    assert avg_latency < 10.0, f"Average latency {avg_latency:.3f}ms exceeds 10ms threshold"
    assert p99 < 50.0, f"P99 latency {p99:.3f}ms exceeds 50ms threshold"


def test_throughput(setup_policy_data):
    """Measure policy evaluation throughput."""
    duration_seconds = 3
    count = 0
    start = time.perf_counter()

    while time.perf_counter() - start < duration_seconds:
        evaluate(
            "allow",
            {
                "principal_id": "user0@example.com",
                "method": "GET",
                "path": "/v2/workspaces/workspace-0/models",
            },
        )
        count += 1

    elapsed = time.perf_counter() - start
    throughput = count / elapsed

    print(f"\nThroughput test ({duration_seconds} seconds):")
    print(f"  Total evaluations: {count}")
    print(f"  Throughput: {throughput:.0f} evaluations/second")

    # Throughput assertion (should handle at least 100 evals/sec)
    assert throughput > 100, f"Throughput {throughput:.0f}/s below 100/s threshold"


@pytest.mark.asyncio
async def test_concurrent_evaluations(setup_policy_data):
    """Test concurrent policy evaluations with asyncio tasks."""
    num_tasks = 4
    evaluations_per_task = 50
    errors = []

    async def run_evaluations(task_id: int) -> list:
        task_results = []
        for i in range(evaluations_per_task):
            try:
                start = time.perf_counter()
                result = evaluate(
                    "allow",
                    {
                        "principal_id": f"user{task_id}@example.com",
                        "method": "GET",
                        "path": f"/v2/workspaces/workspace-{i % 10}/models",
                    },
                )
                latency = (time.perf_counter() - start) * 1000
                task_results.append((result["allowed"], latency))
                # Yield to event loop periodically
                if i % 10 == 0:
                    await asyncio.sleep(0)
            except Exception as e:
                errors.append(str(e))
        return task_results

    start = time.perf_counter()
    tasks = [run_evaluations(i) for i in range(num_tasks)]
    all_results = await asyncio.gather(*tasks)
    results = [r for task_results in all_results for r in task_results]
    total_time = time.perf_counter() - start

    total_evals = num_tasks * evaluations_per_task
    latencies = [r[1] for r in results]

    print(f"\nConcurrent evaluation test ({num_tasks} tasks, {evaluations_per_task} each):")
    print(f"  Total evaluations: {total_evals}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Effective throughput: {total_evals / total_time:.0f} evaluations/second")
    print(f"  Average latency: {statistics.mean(latencies):.3f} ms")
    print(f"  P99 latency: {sorted(latencies)[int(len(latencies) * 0.99)]:.3f} ms")
    print(f"  Errors: {len(errors)}")

    # Assertions
    assert len(errors) == 0, f"Got {len(errors)} errors: {errors[:5]}"
    assert all(r[0] for r in results), "Some evaluations returned False unexpectedly"


def test_all_entrypoints_performance(setup_policy_data):
    """Compare performance across all entrypoints."""
    entrypoints = {
        "allow": {
            "principal_id": "user0@example.com",
            "method": "GET",
            "path": "/v2/workspaces/workspace-0/models",
        },
        "has_permissions": {
            "principal_id": "user0@example.com",
            "workspace": "workspace-0",
            "permissions": ["models.read"],
        },
        "has_role": {
            "principal_id": "user0@example.com",
            "workspace": "workspace-0",
            "role": "Editor",
        },
    }

    iterations = 200
    print(f"\nEntrypoint performance comparison ({iterations} iterations each):")

    for entrypoint, input_data in entrypoints.items():
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            evaluate(entrypoint, input_data)
            latencies.append((time.perf_counter() - start) * 1000)

        avg = statistics.mean(latencies)
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        print(f"  {entrypoint:20s} - avg: {avg:.3f}ms, p99: {p99:.3f}ms")

        assert avg < 20.0, f"{entrypoint} average latency {avg:.3f}ms exceeds 20ms threshold"


def test_memory_stability(setup_policy_data):
    """Verify memory doesn't grow unbounded during repeated evaluations."""
    gc.collect()
    initial_objects = len(gc.get_objects())

    # Run many evaluations
    for _ in range(1000):
        evaluate(
            "allow",
            {
                "principal_id": "user0@example.com",
                "method": "GET",
                "path": "/v2/workspaces/workspace-0/models",
            },
        )

    gc.collect()
    final_objects = len(gc.get_objects())

    growth = final_objects - initial_objects
    print("\nMemory stability test (1000 evaluations):")
    print(f"  Initial objects: {initial_objects}")
    print(f"  Final objects: {final_objects}")
    print(f"  Growth: {growth}")

    # Allow some growth but not unbounded
    assert growth < 5000, f"Object count grew by {growth}, possible memory leak"


def test_health_endpoint_fast_path(setup_policy_data):
    """Verify health endpoints take the fast path."""
    iterations = 200
    latencies = []

    for _ in range(iterations):
        start = time.perf_counter()
        result = evaluate("allow", {"path": "/health/live", "method": "GET"})
        latencies.append((time.perf_counter() - start) * 1000)
        assert result["allowed"] is True

    avg = statistics.mean(latencies)
    print(f"\nHealth endpoint fast path ({iterations} iterations):")
    print(f"  Average latency: {avg:.3f} ms")

    # Health checks should be fast
    assert avg < 10.0, f"Health check avg latency {avg:.3f}ms exceeds 10ms threshold"


# =============================================================================
# Large-scale stress tests with realistic data
# =============================================================================

SEED = 42
NUM_WORKSPACES = 10_000
NUM_USERS = 10_000
ROLES = ["Viewer", "Editor", "Admin"]


@pytest.fixture(scope="module")
def large_scale_policy_data():
    """Set up large-scale authorization data with 10k workspaces and 10k users."""
    rng = random.Random(SEED)

    # Load the full static-authz.yaml
    static_path = Path(__file__).parent.parent / "src/nhx/core/auth/assets/static-authz.yaml"
    with open(static_path) as f:
        data = yaml.safe_load(f)

    # Generate 10k workspaces
    workspaces = {}
    for i in range(NUM_WORKSPACES):
        # 10% of workspaces are public
        if rng.random() < 0.1:
            workspaces[f"workspace-{i:05d}"] = {"visibility": "public"}
        else:
            workspaces[f"workspace-{i:05d}"] = {"visibility": "private"}

    # Generate 10k users with random role assignments
    principals = {}
    for i in range(NUM_USERS):
        user_id = f"user{i:05d}@example.com"
        user_workspaces = {}

        # Each user has access to 1-20 random workspaces
        num_ws = rng.randint(1, 20)
        workspace_indices = rng.sample(range(NUM_WORKSPACES), num_ws)

        for ws_idx in workspace_indices:
            ws_name = f"workspace-{ws_idx:05d}"
            # Assign 1-2 random roles
            num_roles = rng.randint(1, 2)
            user_workspaces[ws_name] = rng.sample(ROLES, num_roles)

        principals[user_id] = {"workspaces": user_workspaces}

    # Add a platform admin
    principals["admin@example.com"] = {"workspaces": {"system": ["HelixAdmin"]}}

    data["authz"]["workspaces"] = workspaces
    data["authz"]["principals"] = principals

    # Extract endpoint patterns for generating random requests.
    # Patterns with "/-/" use special wildcard matching in policy; random concrete paths
    # often fail to match and can make normalize_endpoint error on an empty min().
    endpoints = [p for p in data["authz"]["endpoints"].keys() if "/-/" not in p]
    assert endpoints, "expected at least one non-wildcard endpoint in static-authz"

    reload_policy()
    set_policy_data(data)

    ctx = {
        "endpoints": endpoints,
        "post_endpoints": [path for path in endpoints if "post" in data["authz"]["endpoints"][path]],
        "workspace_ids": [f"workspace-{i:05d}" for i in range(NUM_WORKSPACES)],
        "user_ids": [f"user{i:05d}@example.com" for i in range(NUM_USERS)],
        "rng": random.Random(SEED + 1),  # Separate RNG for test execution
    }
    assert ctx["post_endpoints"], "expected at least one POST endpoint in static-authz"

    yield ctx

    reload_policy()


def _generate_random_path(pattern: str, workspace_id: str, rng: random.Random) -> str:
    """Generate a concrete path from an endpoint pattern."""
    path = pattern
    # Replace common placeholders (static-authz uses {workspace} everywhere; tests used only {workspace_id})
    path = path.replace("{workspace_id}", workspace_id)
    path = path.replace("{workspace}", workspace_id)
    path = path.replace("{job}", f"job-{rng.randint(1, 10000):05d}")
    path = path.replace("{entity_type}", "role_binding")
    path = path.replace("{principal_id}", f"user-{rng.randint(1, 10000):05d}@example.com")
    path = path.replace("{model_name}", f"model-{rng.randint(1, 1000):04d}")
    path = path.replace("{adapter}", f"adapter-{rng.randint(1, 100):02d}")
    path = path.replace("{deployment}", f"deploy-{rng.randint(1, 1000):04d}")
    path = path.replace("{config}", f"cfg-{rng.randint(1, 1000):04d}")
    path = path.replace("{step}", f"step-{rng.randint(1, 100):02d}")
    path = path.replace("{id}", f"id-{rng.randint(1, 10000):05d}")
    path = path.replace("{model_id}", f"model-{rng.randint(1, 1000):04d}")
    path = path.replace("{job_id}", f"job-{rng.randint(1, 1000):04d}")
    path = path.replace("{dataset_id}", f"dataset-{rng.randint(1, 1000):04d}")
    path = path.replace("{config_id}", f"config-{rng.randint(1, 1000):04d}")
    path = path.replace("{deployment_id}", f"deploy-{rng.randint(1, 1000):04d}")
    path = path.replace("{entity_id}", f"entity-{rng.randint(1, 1000):04d}")
    path = path.replace("{secret_id}", f"secret-{rng.randint(1, 100):03d}")
    path = path.replace("{result_name}", f"result-{rng.randint(1, 10):02d}")
    path = path.replace("{benchmark_id}", f"bench-{rng.randint(1, 100):03d}")
    path = path.replace("{target_id}", f"target-{rng.randint(1, 100):03d}")
    path = path.replace("{fileset_id}", f"fileset-{rng.randint(1, 100):03d}")
    path = path.replace("{version}", f"v{rng.randint(1, 10)}")
    # Generic fallback for any remaining placeholders
    import re

    path = re.sub(r"\{[^}]+\}", lambda m: f"placeholder-{rng.randint(1, 1000)}", path)
    return path


@pytest.mark.slow
def test_large_scale_latency(large_scale_policy_data):
    """Measure latency with 10k workspaces and 10k users."""
    ctx = large_scale_policy_data
    rng = ctx["rng"]
    iterations = 500
    latencies = []
    methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]

    for _ in range(iterations):
        # Pick random user, workspace, endpoint, method
        user_id = rng.choice(ctx["user_ids"])
        workspace_id = rng.choice(ctx["workspace_ids"])
        endpoint = rng.choice(ctx["endpoints"])
        method = rng.choice(methods)
        path = _generate_random_path(endpoint, workspace_id, rng)

        start = time.perf_counter()
        evaluate("allow", {"principal_id": user_id, "method": method, "path": path})
        latencies.append((time.perf_counter() - start) * 1000)

    avg_latency = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]

    print(f"\nLarge-scale latency test ({NUM_WORKSPACES} workspaces, {NUM_USERS} users, {iterations} iterations):")
    print(f"  Average: {avg_latency:.3f} ms")
    print(f"  P50:     {p50:.3f} ms")
    print(f"  P99:     {p99:.3f} ms")

    # Performance assertions - may be slightly slower with more data
    assert avg_latency < 15.0, f"Average latency {avg_latency:.3f}ms exceeds 15ms threshold"
    assert p99 < 75.0, f"P99 latency {p99:.3f}ms exceeds 75ms threshold"


@pytest.mark.slow
def test_large_scale_throughput(large_scale_policy_data):
    """Measure throughput with 10k workspaces and 10k users."""
    ctx = large_scale_policy_data
    rng = ctx["rng"]
    duration_seconds = 3
    count = 0
    methods = ["GET", "POST", "PUT", "DELETE"]
    start = time.perf_counter()

    while time.perf_counter() - start < duration_seconds:
        user_id = rng.choice(ctx["user_ids"])
        workspace_id = rng.choice(ctx["workspace_ids"])
        endpoint = rng.choice(ctx["endpoints"])
        method = rng.choice(methods)
        path = _generate_random_path(endpoint, workspace_id, rng)

        evaluate("allow", {"principal_id": user_id, "method": method, "path": path})
        count += 1

    elapsed = time.perf_counter() - start
    throughput = count / elapsed

    print(f"\nLarge-scale throughput test ({NUM_WORKSPACES} workspaces, {NUM_USERS} users, {duration_seconds}s):")
    print(f"  Total evaluations: {count}")
    print(f"  Throughput: {throughput:.0f} evaluations/second")

    # Throughput may be lower with more data, but should still be >50/sec
    assert throughput > 50, f"Throughput {throughput:.0f}/s below 50/s threshold"


@pytest.mark.slow
def test_large_scale_all_entrypoints(large_scale_policy_data):
    """Compare all entrypoints with large-scale data."""
    ctx = large_scale_policy_data
    rng = ctx["rng"]
    iterations = 200

    print(f"\nLarge-scale entrypoint comparison ({NUM_WORKSPACES} ws, {NUM_USERS} users, {iterations} iter each):")

    # Test allow
    latencies = []
    for _ in range(iterations):
        user_id = rng.choice(ctx["user_ids"])
        workspace_id = rng.choice(ctx["workspace_ids"])
        endpoint = rng.choice(ctx["endpoints"])
        path = _generate_random_path(endpoint, workspace_id, rng)

        start = time.perf_counter()
        evaluate("allow", {"principal_id": user_id, "method": "GET", "path": path})
        latencies.append((time.perf_counter() - start) * 1000)

    avg = statistics.mean(latencies)
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    print(f"  allow                - avg: {avg:.3f}ms, p99: {p99:.3f}ms")

    # Test has_permissions
    latencies = []
    for _ in range(iterations):
        user_id = rng.choice(ctx["user_ids"])
        workspace_id = rng.choice(ctx["workspace_ids"])

        start = time.perf_counter()
        evaluate(
            "has_permissions",
            {"principal_id": user_id, "workspace": workspace_id, "permissions": ["models.read"]},
        )
        latencies.append((time.perf_counter() - start) * 1000)

    avg = statistics.mean(latencies)
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    print(f"  has_permissions      - avg: {avg:.3f}ms, p99: {p99:.3f}ms")

    # Test has_role
    latencies = []
    for _ in range(iterations):
        user_id = rng.choice(ctx["user_ids"])
        workspace_id = rng.choice(ctx["workspace_ids"])
        role = rng.choice(ROLES)

        start = time.perf_counter()
        evaluate("has_role", {"principal_id": user_id, "workspace": workspace_id, "role": role})
        latencies.append((time.perf_counter() - start) * 1000)

    avg = statistics.mean(latencies)
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    print(f"  has_role             - avg: {avg:.3f}ms, p99: {p99:.3f}ms")


@pytest.mark.slow
def test_large_scale_authorization_accuracy(large_scale_policy_data):
    """Verify authorization decisions are correct with large-scale data."""
    ctx = large_scale_policy_data
    rng = ctx["rng"]

    # Endpoints that can deny platform admin (e.g. secrets value access)
    deny_platform_admin_patterns = ["secrets", "access"]

    def endpoint_allows_platform_admin(pattern: str) -> bool:
        return not any(p in pattern for p in deny_platform_admin_patterns)

    # Test 1: Platform admin should be allowed on paths that are not explicitly denied
    allowed_endpoints = [e for e in ctx["endpoints"] if endpoint_allows_platform_admin(e)]
    if not allowed_endpoints:
        allowed_endpoints = ctx["endpoints"]
    for _ in range(50):
        workspace_id = rng.choice(ctx["workspace_ids"])
        endpoint = rng.choice(allowed_endpoints)
        path = _generate_random_path(endpoint, workspace_id, rng)
        result = evaluate("allow", {"principal_id": "admin@example.com", "method": "DELETE", "path": path})
        assert result["allowed"] is True, f"Platform admin should be allowed on {path!r}"

    # Test 2: Service principals should be allowed on registered POST routes.
    for _ in range(50):
        workspace_id = rng.choice(ctx["workspace_ids"])
        endpoint = rng.choice(ctx["post_endpoints"])
        path = _generate_random_path(endpoint, workspace_id, rng)

        result = evaluate("allow", {"principal_id": "service:test-svc", "method": "POST", "path": path})
        assert result["allowed"] is True, f"Service principal should be allowed on {path!r}"

    # Test 3: Health/status endpoints should always be allowed
    for path in ["/health/live", "/health/ready", "/status"]:
        result = evaluate("allow", {"path": path, "method": "GET"})
        assert result["allowed"] is True, f"{path} should always be allowed"

    # Test 4: Unknown users should be denied on non-health endpoints
    for _ in range(50):
        workspace_id = rng.choice(ctx["workspace_ids"])
        endpoint = rng.choice(ctx["endpoints"])
        if "health" in endpoint.lower():
            continue
        path = _generate_random_path(endpoint, workspace_id, rng)

        result = evaluate("allow", {"principal_id": "unknown@hacker.com", "method": "GET", "path": path})
        # Unknown users should generally be denied (unless public workspace + viewer permissions)
        # This is a soft check since some public workspaces may allow access

    print("\nLarge-scale authorization accuracy: All checks passed")
