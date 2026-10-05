# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DockerGPUPool."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from nhx.common.docker.gpu_pool import DockerGPUPool, GPUAllocationError, GPUPoolStatus


class TestDockerGPUPoolInit:
    """Tests for GPU pool initialization."""

    @pytest.mark.parametrize(
        "device_ids,expected_count,expected_keys",
        [
            pytest.param([0], 1, {0}, id="single_gpu"),
            pytest.param([0, 1, 2, 3], 4, {0, 1, 2, 3}, id="multiple_gpus"),
            pytest.param([0, 2, 5, 7], 4, {0, 2, 5, 7}, id="non_contiguous"),
            pytest.param([], 0, set(), id="empty"),
            pytest.param([99], 1, {99}, id="high_device_id"),
        ],
    )
    def test_init_with_device_ids(self, device_ids, expected_count, expected_keys):
        """Test initialization with various GPU device ID configurations."""
        pool = DockerGPUPool(reserved_gpu_device_ids=device_ids)
        assert pool.num_reserved_gpus == expected_count
        assert set(pool.gpu_to_workload_id.keys()) == expected_keys
        # All GPUs should be initially unallocated
        assert all(v is None for v in pool.gpu_to_workload_id.values())


class TestDockerGPUPoolAllocation:
    """Tests for GPU allocation."""

    @pytest.mark.parametrize(
        "pool_size,num_requested",
        [
            pytest.param(4, 1, id="single_from_large_pool"),
            pytest.param(4, 2, id="multiple_from_large_pool"),
            pytest.param(4, 4, id="all_from_pool"),
            pytest.param(1, 1, id="single_from_single"),
            pytest.param(8, 3, id="three_from_eight"),
        ],
    )
    def test_allocate_gpus(self, pool_size, num_requested):
        """Test allocating various numbers of GPUs from pools of various sizes."""
        device_ids = list(range(pool_size))
        pool = DockerGPUPool(reserved_gpu_device_ids=device_ids)

        gpu_ids = pool.allocate_gpu("workload-1", num_requested=num_requested)

        assert len(gpu_ids) == num_requested
        assert len(set(gpu_ids)) == num_requested  # All unique
        assert all(gpu_id in device_ids for gpu_id in gpu_ids)
        for gpu_id in gpu_ids:
            assert pool.gpu_to_workload_id[gpu_id] == "workload-1"
        assert pool.get_available_count() == pool_size - num_requested

    def test_allocate_sequential_workloads(self):
        """Test allocating GPUs for multiple sequential workloads without overlap."""
        pool = DockerGPUPool(reserved_gpu_device_ids=[0, 1, 2, 3])

        gpu_ids_1 = pool.allocate_gpu("workload-1", num_requested=2)
        gpu_ids_2 = pool.allocate_gpu("workload-2", num_requested=2)

        assert len(gpu_ids_1) == 2
        assert len(gpu_ids_2) == 2
        assert set(gpu_ids_1).isdisjoint(set(gpu_ids_2))

    @pytest.mark.parametrize(
        "pool_size,pre_allocate,num_requested,expected_error_fragment,expected_transient",
        [
            pytest.param(2, 1, 2, "Requested 2", True, id="insufficient_remaining"),
            pytest.param(1, 1, 1, "Requested 1", True, id="none_available"),
            pytest.param(1, 0, 2, "Requested 2", False, id="request_exceeds_total"),
            pytest.param(0, 0, 1, "Requested 1", False, id="empty_pool"),
        ],
    )
    def test_allocate_raises_when_insufficient(
        self, pool_size, pre_allocate, num_requested, expected_error_fragment, expected_transient
    ):
        """Test that allocation raises GPUAllocationError when not enough GPUs available."""
        pool = DockerGPUPool(reserved_gpu_device_ids=list(range(pool_size)))
        if pre_allocate > 0:
            pool.allocate_gpu("pre-existing", num_requested=pre_allocate)

        with pytest.raises(GPUAllocationError) as exc_info:
            pool.allocate_gpu("workload-new", num_requested=num_requested)

        assert expected_error_fragment in str(exc_info.value)
        assert exc_info.value.requested == num_requested
        assert exc_info.value.available == pool_size - pre_allocate
        assert exc_info.value.total == pool_size
        assert exc_info.value.is_transient_capacity_exhaustion is expected_transient

    @pytest.mark.parametrize(
        "invalid_value",
        [
            pytest.param(0, id="zero"),
            pytest.param(-1, id="negative_one"),
            pytest.param(-2, id="negative_two"),
            pytest.param(-10, id="large_negative"),
        ],
    )
    def test_allocate_raises_on_invalid_num_requested(self, invalid_value):
        """Test that invalid num_requested values (zero or negative) raise GPUAllocationError."""
        pool = DockerGPUPool(reserved_gpu_device_ids=[0, 1, 2, 3])

        with pytest.raises(GPUAllocationError) as exc_info:
            pool.allocate_gpu("workload-invalid", num_requested=invalid_value)

        assert "Invalid GPU request" in str(exc_info.value)
        assert "Must be a positive integer" in str(exc_info.value)
        assert exc_info.value.is_transient_capacity_exhaustion is False


class TestDockerGPUPoolRelease:
    """Tests for GPU release."""

    @pytest.mark.parametrize(
        "pool_size,num_allocated",
        [
            pytest.param(2, 1, id="release_one"),
            pytest.param(4, 3, id="release_multiple"),
            pytest.param(4, 4, id="release_all"),
        ],
    )
    def test_release_gpus(self, pool_size, num_allocated):
        """Test releasing GPUs restores availability."""
        pool = DockerGPUPool(reserved_gpu_device_ids=list(range(pool_size)))
        gpu_ids = pool.allocate_gpu("workload-1", num_requested=num_allocated)

        released = pool.release_gpu("workload-1")

        assert set(released) == set(gpu_ids)
        assert pool.get_available_count() == pool_size
        for gpu_id in gpu_ids:
            assert pool.gpu_to_workload_id[gpu_id] is None

    def test_release_nonexistent_workload_is_noop(self):
        """Test releasing GPUs for a workload that doesn't exist returns empty list."""
        pool = DockerGPUPool(reserved_gpu_device_ids=[0, 1])
        pool.allocate_gpu("workload-1", num_requested=1)

        released = pool.release_gpu("nonexistent-workload")

        assert released == []
        assert pool.get_available_count() == 1  # Original allocation still in place

    def test_release_already_released_is_noop(self):
        """Test releasing GPUs that were already released returns empty list."""
        pool = DockerGPUPool(reserved_gpu_device_ids=[0])
        pool.allocate_gpu("workload-1", num_requested=1)
        pool.release_gpu("workload-1")

        released = pool.release_gpu("workload-1")

        assert released == []

    def test_release_enables_reallocation(self):
        """Test that released GPUs can be reallocated to new workloads."""
        pool = DockerGPUPool(reserved_gpu_device_ids=[0])
        pool.allocate_gpu("workload-1", num_requested=1)

        with pytest.raises(GPUAllocationError):
            pool.allocate_gpu("workload-2", num_requested=1)

        pool.release_gpu("workload-1")
        gpu_ids = pool.allocate_gpu("workload-2", num_requested=1)

        assert gpu_ids == [0]
        assert pool.gpu_to_workload_id[0] == "workload-2"


class TestDockerGPUPoolHelperMethods:
    """Tests for helper methods."""

    @pytest.mark.parametrize(
        "pool_size,allocations,expected_available",
        [
            pytest.param(4, [], 4, id="all_available"),
            pytest.param(4, [("w1", 2)], 2, id="partial_allocation"),
            pytest.param(4, [("w1", 2), ("w2", 2)], 0, id="fully_allocated"),
            pytest.param(2, [("w1", 1)], 1, id="half_allocated"),
            pytest.param(0, [], 0, id="empty_pool"),
        ],
    )
    def test_get_available_count(self, pool_size, allocations, expected_available):
        """Test available count under various allocation states."""
        pool = DockerGPUPool(reserved_gpu_device_ids=list(range(pool_size)))
        for workload_id, num in allocations:
            pool.allocate_gpu(workload_id, num_requested=num)

        assert pool.get_available_count() == expected_available

    def test_get_allocated_workloads_empty_when_none_allocated(self):
        """Test getting allocated workloads returns empty dict when none allocated."""
        pool = DockerGPUPool(reserved_gpu_device_ids=[0, 1])
        assert pool.get_allocated_workloads() == {}

    def test_get_allocated_workloads_reflects_allocations(self):
        """Test getting allocated workloads returns correct mapping."""
        pool = DockerGPUPool(reserved_gpu_device_ids=[0, 1, 2, 3])
        gpu_ids_1 = pool.allocate_gpu("workload-1", num_requested=2)
        gpu_ids_2 = pool.allocate_gpu("workload-2", num_requested=1)

        allocated = pool.get_allocated_workloads()

        assert len(allocated) == 3
        for gpu_id in gpu_ids_1:
            assert allocated[gpu_id] == "workload-1"
        for gpu_id in gpu_ids_2:
            assert allocated[gpu_id] == "workload-2"


class TestDockerGPUPoolDiagnostics:
    """Tests for diagnostic methods."""

    def test_get_pool_status_returns_dataclass(self):
        """Test that get_pool_status returns a GPUPoolStatus dataclass."""
        pool = DockerGPUPool(reserved_gpu_device_ids=[0, 1])
        status = pool.get_pool_status()
        assert isinstance(status, GPUPoolStatus)

    def test_get_pool_status_empty_pool(self):
        """Test get_pool_status returns correct info for empty pool."""
        pool = DockerGPUPool(reserved_gpu_device_ids=[])
        status = pool.get_pool_status()

        assert status.total == 0
        assert status.available == 0
        assert status.allocated == 0
        assert status.allocations == {}
        assert status.gpu_state == {}

    def test_get_pool_status_all_available(self):
        """Test get_pool_status when all GPUs are available."""
        pool = DockerGPUPool(reserved_gpu_device_ids=[0, 1, 2, 3])
        status = pool.get_pool_status()

        assert status.total == 4
        assert status.available == 4
        assert status.allocated == 0
        assert status.allocations == {}
        assert status.gpu_state == {0: None, 1: None, 2: None, 3: None}

    def test_get_pool_status_partial_allocation(self):
        """Test get_pool_status with some GPUs allocated."""
        pool = DockerGPUPool(reserved_gpu_device_ids=[0, 1, 2, 3])
        gpu_ids = pool.allocate_gpu("workload-1", num_requested=2)

        status = pool.get_pool_status()

        assert status.total == 4
        assert status.available == 2
        assert status.allocated == 2
        assert "workload-1" in status.allocations
        assert set(status.allocations["workload-1"]) == set(gpu_ids)

    def test_get_pool_status_multiple_workloads(self):
        """Test get_pool_status with multiple workloads."""
        pool = DockerGPUPool(reserved_gpu_device_ids=[0, 1, 2, 3])
        gpu_ids_1 = pool.allocate_gpu("job-123", num_requested=2)
        gpu_ids_2 = pool.allocate_gpu("deployment-abc", num_requested=1)

        status = pool.get_pool_status()

        assert status.total == 4
        assert status.available == 1
        assert status.allocated == 3
        assert set(status.allocations["job-123"]) == set(gpu_ids_1)
        assert set(status.allocations["deployment-abc"]) == set(gpu_ids_2)

    def test_force_release_all_empty_pool(self):
        """Test force_release_all on pool with no allocations."""
        pool = DockerGPUPool(reserved_gpu_device_ids=[0, 1])
        released = pool.force_release_all()

        assert released == {}
        assert pool.get_available_count() == 2

    def test_force_release_all_with_allocations(self):
        """Test force_release_all releases all allocated GPUs."""
        pool = DockerGPUPool(reserved_gpu_device_ids=[0, 1, 2, 3])
        gpu_ids_1 = pool.allocate_gpu("workload-1", num_requested=2)
        gpu_ids_2 = pool.allocate_gpu("workload-2", num_requested=1)

        assert pool.get_available_count() == 1

        released = pool.force_release_all()

        # Check returned dict contains the workloads and their GPU IDs
        assert "workload-1" in released
        assert "workload-2" in released
        assert set(released["workload-1"]) == set(gpu_ids_1)
        assert set(released["workload-2"]) == set(gpu_ids_2)
        # All GPUs should now be available
        assert pool.get_available_count() == 4
        assert pool.get_allocated_workloads() == {}

    def test_force_release_all_then_reallocate(self):
        """Test that GPUs can be reallocated after force_release_all."""
        pool = DockerGPUPool(reserved_gpu_device_ids=[0])
        pool.allocate_gpu("old-workload", num_requested=1)

        pool.force_release_all()
        gpu_ids = pool.allocate_gpu("new-workload", num_requested=1)

        assert gpu_ids == [0]
        assert pool.gpu_to_workload_id[0] == "new-workload"


class TestDockerGPUPoolThreadSafety:
    """Tests for thread safety."""

    @pytest.mark.parametrize(
        "pool_size,num_workers,num_allocations",
        [
            pytest.param(100, 50, 100, id="large_pool_high_contention"),
            pytest.param(10, 10, 10, id="exact_fit"),
            pytest.param(50, 25, 50, id="medium_pool"),
        ],
    )
    def test_concurrent_allocations_all_succeed(self, pool_size, num_workers, num_allocations):
        """Test that concurrent allocations succeed when pool has enough capacity."""
        pool = DockerGPUPool(reserved_gpu_device_ids=list(range(pool_size)))
        results = []
        errors = []
        lock = threading.Lock()

        def allocate_one(workload_id: str):
            try:
                gpu_ids = pool.allocate_gpu(workload_id, num_requested=1)
                with lock:
                    results.append((workload_id, gpu_ids))
            except GPUAllocationError as e:
                with lock:
                    errors.append((workload_id, e))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(allocate_one, f"workload-{i}") for i in range(num_allocations)]
            for future in futures:
                future.result()

        assert len(results) == num_allocations
        assert len(errors) == 0
        all_gpu_ids = [gpu_id for _, gpu_ids in results for gpu_id in gpu_ids]
        assert len(set(all_gpu_ids)) == num_allocations  # All unique

    @pytest.mark.parametrize(
        "pool_size,num_requests,expected_success,expected_failure",
        [
            pytest.param(4, 10, 4, 6, id="4_gpus_10_requests"),
            pytest.param(2, 5, 2, 3, id="2_gpus_5_requests"),
            pytest.param(1, 3, 1, 2, id="1_gpu_3_requests"),
        ],
    )
    def test_concurrent_allocations_with_contention(self, pool_size, num_requests, expected_success, expected_failure):
        """Test concurrent allocations when demand exceeds supply."""
        pool = DockerGPUPool(reserved_gpu_device_ids=list(range(pool_size)))
        results = []
        errors = []
        lock = threading.Lock()

        def allocate_one(workload_id: str):
            try:
                gpu_ids = pool.allocate_gpu(workload_id, num_requested=1)
                with lock:
                    results.append((workload_id, gpu_ids))
            except GPUAllocationError as e:
                with lock:
                    errors.append((workload_id, e))

        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(allocate_one, f"workload-{i}") for i in range(num_requests)]
            for future in futures:
                future.result()

        assert len(results) == expected_success
        assert len(errors) == expected_failure

    def test_concurrent_allocate_and_release_restores_pool(self):
        """Test concurrent allocations and releases leave pool in clean state."""
        pool = DockerGPUPool(reserved_gpu_device_ids=[0, 1])

        def allocate_and_release(workload_id: str):
            try:
                pool.allocate_gpu(workload_id, num_requested=1)
                threading.Event().wait(0.001)  # Simulate work
                pool.release_gpu(workload_id)
            except GPUAllocationError:
                pass

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(allocate_and_release, f"workload-{i}") for i in range(100)]
            for future in futures:
                future.result()

        assert pool.get_available_count() == 2
        assert pool.get_allocated_workloads() == {}
