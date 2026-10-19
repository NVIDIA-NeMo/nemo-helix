# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the quickstart container module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nemo_helix_ext.quickstart.config import QuickstartConfig
from nemo_helix_ext.quickstart.container import ContainerManager
from pydantic import SecretStr


class TestHasFloatingTag:
    """Tests for ContainerManager._has_floating_tag."""

    def _make_manager(self, image: str) -> ContainerManager:
        config = MagicMock()
        config.image = image
        manager = ContainerManager(config)
        return manager

    @pytest.mark.parametrize(
        "image",
        [
            pytest.param("nginx:latest", id="simple_latest"),
            pytest.param("nvcr.io/nvidia/nhx-api:latest", id="registry_latest"),
            pytest.param(
                "registry.example.com:443/nvidia-nemo/nhx-api:latest",
                id="registry_with_port_latest",
            ),
            pytest.param("nginx", id="no_tag_implicit_latest"),
            pytest.param("nvcr.io/nvidia/nhx-api", id="registry_no_tag"),
        ],
    )
    def test_floating_tags(self, image: str):
        assert self._make_manager(image)._has_floating_tag() is True

    @pytest.mark.parametrize(
        "image",
        [
            pytest.param("nginx:1.25", id="pinned_version"),
            pytest.param("nvcr.io/nvidia/nhx-api:25.01", id="registry_pinned"),
            pytest.param(
                "registry.example.com:443/nvidia-nemo/nhx-api:v2.0.0",
                id="registry_with_port_pinned",
            ),
            pytest.param(
                "myimage:sha-abc1234",
                id="sha_prefix_tag",
            ),
        ],
    )
    def test_non_floating_tags(self, image: str):
        assert self._make_manager(image)._has_floating_tag() is False


class TestDetectHostGPUDeviceIds:
    """Tests for ContainerManager._detect_host_gpu_device_ids."""

    def test_detect_single_gpu(self):
        """Test detection with a single GPU."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = ContainerManager._detect_host_gpu_device_ids()

        assert result == [0]
        mock_pynvml.nvmlInit.assert_called_once()
        mock_pynvml.nvmlShutdown.assert_called_once()

    @pytest.mark.parametrize(
        "gpu_count,expected_ids",
        [
            pytest.param(2, [0, 1], id="two_gpus"),
            pytest.param(4, [0, 1, 2, 3], id="four_gpus"),
            pytest.param(8, [0, 1, 2, 3, 4, 5, 6, 7], id="eight_gpus"),
        ],
    )
    def test_detect_multiple_gpus(self, gpu_count, expected_ids):
        """Test detection with multiple GPUs."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = gpu_count

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = ContainerManager._detect_host_gpu_device_ids()

        assert result == expected_ids

    def test_returns_none_when_no_gpus(self):
        """Test that detection returns None when zero GPUs are reported."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 0

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = ContainerManager._detect_host_gpu_device_ids()

        assert result is None

    def test_returns_none_when_gpu_detection_unavailable(self):
        """Test that detection returns None when the GPU detection library is not installed."""
        with patch.dict("sys.modules", {"pynvml": None}):
            result = ContainerManager._detect_host_gpu_device_ids()

        assert result is None

    def test_returns_none_on_init_failure(self):
        """Test that detection returns None when initialization fails."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = Exception("Driver not loaded")

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = ContainerManager._detect_host_gpu_device_ids()

        assert result is None

    def test_returns_none_on_count_failure(self):
        """Test that detection returns None when getting device count fails."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.side_effect = Exception("Query failed")

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = ContainerManager._detect_host_gpu_device_ids()

        assert result is None


class TestCreateEnvironmentGPU:
    """Tests for GPU env var passthrough in _create_environment.

    GPU device IDs come from config.reserved_gpu_device_ids (set at configure time).
    No host detection at up time.
    """

    def _make_manager(
        self,
        use_gpu: bool = False,
        reserved_gpu_device_ids: str | None = None,
    ) -> ContainerManager:
        """Create a ContainerManager with a minimal config."""
        config = MagicMock()
        config.use_gpu = use_gpu
        config.reserved_gpu_device_ids = reserved_gpu_device_ids
        config.ngc_api_key = None
        config.network_name = "test-network"
        config.container_name = "test-container"
        config.platform_config_path = None
        config.auth_enabled = False
        config.image = "my-registry/nhx-api:local"
        config.parse_image_components.return_value = ("my-registry", "local")
        manager = ContainerManager.__new__(ContainerManager)
        manager.config = config
        return manager

    def test_gpu_ids_passed_when_use_gpu_and_reserved_ids_set(self):
        """Test that GPU device IDs from config are passed when use_gpu is True."""
        manager = self._make_manager(use_gpu=True, reserved_gpu_device_ids="0,1")
        mock_platform_config = MagicMock()
        mock_platform_config.to_env_vars.return_value = {}

        env = manager._create_environment(mock_platform_config)

        assert env["NHX_DOCKER_RESERVED_GPU_DEVICE_IDS"] == "0,1"

    def test_gpu_ids_not_passed_when_use_gpu_disabled(self):
        """Test that GPU device IDs are not passed when use_gpu is False."""
        manager = self._make_manager(use_gpu=False, reserved_gpu_device_ids="0,1")
        mock_platform_config = MagicMock()
        mock_platform_config.to_env_vars.return_value = {}

        env = manager._create_environment(mock_platform_config)

        assert "NHX_DOCKER_RESERVED_GPU_DEVICE_IDS" not in env

    def test_gpu_ids_not_passed_when_use_gpu_but_reserved_ids_none(self):
        """Test that env var is not set when use_gpu is True but reserved_gpu_device_ids is None."""
        manager = self._make_manager(use_gpu=True, reserved_gpu_device_ids=None)
        mock_platform_config = MagicMock()
        mock_platform_config.to_env_vars.return_value = {}

        env = manager._create_environment(mock_platform_config)

        assert "NHX_DOCKER_RESERVED_GPU_DEVICE_IDS" not in env

    def test_empty_string_passed_when_reserved_ids_empty(self):
        """Test that empty string is passed when reserved_gpu_device_ids is '' (no GPUs)."""
        manager = self._make_manager(use_gpu=True, reserved_gpu_device_ids="")
        mock_platform_config = MagicMock()
        mock_platform_config.to_env_vars.return_value = {}

        env = manager._create_environment(mock_platform_config)

        assert env["NHX_DOCKER_RESERVED_GPU_DEVICE_IDS"] == ""

    def test_single_gpu_id_format(self):
        """Test that a single GPU ID string is passed through as-is."""
        manager = self._make_manager(use_gpu=True, reserved_gpu_device_ids="0")
        mock_platform_config = MagicMock()
        mock_platform_config.to_env_vars.return_value = {}

        env = manager._create_environment(mock_platform_config)

        assert env["NHX_DOCKER_RESERVED_GPU_DEVICE_IDS"] == "0"

    def test_deployments_docker_backend_uses_quickstart_network_endpoint_mode(self):
        manager = self._make_manager()
        mock_platform_config = MagicMock()
        mock_platform_config.to_env_vars.return_value = {}

        env = manager._create_environment(mock_platform_config)

        assert env["NEMO_DEPLOYMENTS_DOCKER_NETWORK"] == "test-network"
        assert env["NEMO_DEPLOYMENTS_DOCKER_ENDPOINT_MODE"] == "network"

    def test_explicit_platform_config_secret_env_vars_are_preserved(self):
        manager = self._make_manager()
        mock_platform_config = MagicMock()
        mock_platform_config.to_env_vars.return_value = {
            "NHX_SECRETS_ALLOW_KEY_CREATION": "0",
            "NHX_SECRETS_LOCAL_KEY_CREATION_PATH": "/custom/key.txt",
        }

        env = manager._create_environment(mock_platform_config)

        assert env["NHX_SECRETS_ALLOW_KEY_CREATION"] == "0"
        assert env["NHX_SECRETS_LOCAL_KEY_CREATION_PATH"] == "/custom/key.txt"


class TestCreateEnvironmentRegistryCredentials:
    """Tests for registry credential env var passthrough."""

    def _environment_for_config(self, config: QuickstartConfig) -> dict[str, str]:
        manager = ContainerManager.__new__(ContainerManager)
        manager.config = config
        mock_platform_config = MagicMock()
        mock_platform_config.to_env_vars.return_value = {}
        return manager._create_environment(mock_platform_config)

    def test_user_pass_registry_credentials_are_passed_to_jobs_backend(self):
        config = QuickstartConfig(
            image="ghcr.io/nvidia-nemo/nemo/nhx-api:latest",
            registry_host="ghcr.io",
            registry_username="test-user",
            registry_password=SecretStr("test-token"),
        )

        env = self._environment_for_config(config)

        assert env["NHX_IMAGE_REGISTRY"] == "ghcr.io/nvidia-nemo/nemo"
        assert env["NEMO_JOBS_IMAGE_REGISTRY"] == "ghcr.io"
        assert env["NEMO_JOBS_IMAGE_REGISTRY_USER_NAME"] == "test-user"
        assert env["NEMO_JOBS_IMAGE_REGISTRY_PASSWORD"] == "test-token"

    def test_two_part_registry_credentials_are_passed_to_jobs_backend(self):
        config = QuickstartConfig(
            image="docker.io/nhx-api:latest",
            registry_host="docker.io",
            registry_username="test-user",
            registry_password=SecretStr("test-token"),
        )

        env = self._environment_for_config(config)

        assert env["NEMO_JOBS_IMAGE_REGISTRY"] == "docker.io"
        assert env["NEMO_JOBS_IMAGE_REGISTRY_USER_NAME"] == "test-user"
        assert env["NEMO_JOBS_IMAGE_REGISTRY_PASSWORD"] == "test-token"

    def test_user_pass_registry_credentials_are_not_passed_when_host_does_not_match(self):
        config = QuickstartConfig(
            image="ghcr.io/nvidia-nemo/nemo/nhx-api:latest",
            registry_host="registry.example.com",
            registry_username="test-user",
            registry_password=SecretStr("test-token"),
        )

        env = self._environment_for_config(config)

        assert "NEMO_JOBS_IMAGE_REGISTRY" not in env
        assert "NEMO_JOBS_IMAGE_REGISTRY_USER_NAME" not in env
        assert "NEMO_JOBS_IMAGE_REGISTRY_PASSWORD" not in env

    def test_user_pass_registry_env_not_passed_when_credentials_are_missing(self):
        config = QuickstartConfig(
            image="ghcr.io/nvidia-nemo/nemo/nhx-api:latest",
            registry_host="ghcr.io",
            registry_username="test-user",
            registry_password=None,
        )

        env = self._environment_for_config(config)

        assert "NEMO_JOBS_IMAGE_REGISTRY" not in env
        assert "NEMO_JOBS_IMAGE_REGISTRY_USER_NAME" not in env
        assert "NEMO_JOBS_IMAGE_REGISTRY_PASSWORD" not in env

    def test_ngc_registry_env_is_not_passed_when_api_key_is_missing(self):
        config = QuickstartConfig(
            image="nvcr.io/nvidia/platform-api:nightly-20260310",
            ngc_api_key=None,
        )

        env = self._environment_for_config(config)

        assert "NEMO_JOBS_IMAGE_REGISTRY" not in env
        assert "NEMO_JOBS_IMAGE_REGISTRY_USER_NAME" not in env
        assert "NEMO_JOBS_IMAGE_REGISTRY_PASSWORD" not in env

    def test_ngc_registry_credentials_are_passed_to_jobs_backend(self):
        config = QuickstartConfig(
            image="nvcr.io/nvidia/platform-api:nightly-20260310",
            ngc_api_key=SecretStr("ngc-key"),
        )

        env = self._environment_for_config(config)

        assert env["NHX_IMAGE_REGISTRY"] == "nvcr.io/nvidia"
        assert env["NEMO_JOBS_IMAGE_REGISTRY"] == "nvcr.io"
        assert env["NEMO_JOBS_IMAGE_REGISTRY_USER_NAME"] == "$oauthtoken"
        assert env["NEMO_JOBS_IMAGE_REGISTRY_PASSWORD"] == "ngc-key"


class TestContainerManagerStop:
    """Tests for ContainerManager.stop()."""

    def test_stop_lists_containers_by_managed_by_label_and_stops_removes_each(self):
        """Test that stop() lists containers with managed-by=models-controller and stops/removes each."""
        config = MagicMock()
        config.container_name = "nhx-quickstart"
        config.network_name = "nhx-quickstart-network"
        config.save = MagicMock()

        manager = ContainerManager.__new__(ContainerManager)
        manager.config = config

        mock_client = MagicMock()
        mock_c1 = MagicMock()
        mock_c2 = MagicMock()
        mock_client.containers.list.return_value = [mock_c1, mock_c2]
        manager._client = mock_client

        manager._get_container = MagicMock(return_value=None)

        manager.stop(remove=False, timeout=10)

        mock_client.containers.list.assert_called_once_with(
            all=True, filters={"label": ["nhx.nvidia.com/managed-by=models-controller"]}
        )
        mock_c1.stop.assert_called_once_with(timeout=10)
        mock_c1.remove.assert_called_once()
        mock_c2.stop.assert_called_once_with(timeout=10)
        mock_c2.remove.assert_called_once()

    @pytest.mark.parametrize(
        "main_container_present",
        [False, True],
        ids=["main_gone", "main_present"],
    )
    def test_stop_remove_true(self, main_container_present):
        """When remove=True: stop/remove main container (if present), clear config, and remove network."""
        config = MagicMock()
        config.container_name = "nhx-quickstart"
        config.network_name = "nhx-quickstart-network"
        config.save = MagicMock()
        if main_container_present:
            config.container_id = "old-id"

        manager = ContainerManager.__new__(ContainerManager)
        manager.config = config
        manager._client = MagicMock()
        manager._client.containers.list.return_value = []

        mock_main = MagicMock() if main_container_present else None
        manager._get_container = MagicMock(return_value=mock_main)
        manager._remove_existing_network = MagicMock()

        manager.stop(remove=True, timeout=5)

        if main_container_present:
            mock_main.stop.assert_called_once_with(timeout=5)
            mock_main.remove.assert_called_once()
            assert config.container_id is None
            config.save.assert_called_once()
        else:
            config.save.assert_not_called()

        manager._remove_existing_network.assert_called_once()

    def test_stop_remove_false_does_not_remove_network(self):
        """When remove=False, the network should not be touched."""
        config = MagicMock()
        config.container_name = "nhx-quickstart"
        config.network_name = "nhx-quickstart-network"

        manager = ContainerManager.__new__(ContainerManager)
        manager.config = config
        manager._client = MagicMock()
        manager._client.containers.list.return_value = []
        manager._get_container = MagicMock(return_value=None)
        manager._remove_existing_network = MagicMock()

        manager.stop(remove=False, timeout=5)

        manager._remove_existing_network.assert_not_called()


class TestRemoveExistingNetwork:
    """Tests for ContainerManager._remove_existing_network()."""

    def _make_manager(self) -> ContainerManager:
        config = MagicMock()
        config.network_name = "nhx-quickstart-network"
        manager = ContainerManager.__new__(ContainerManager)
        manager.config = config
        manager._client = MagicMock()
        return manager

    def test_disconnects_containers_before_removing_network(self):
        """Containers attached to the network are disconnected before network.remove()."""
        manager = self._make_manager()
        mock_c1 = MagicMock(name="nim-container-1")
        mock_c2 = MagicMock(name="nim-container-2")
        mock_network = MagicMock()
        mock_network.containers = [mock_c1, mock_c2]
        manager._client.networks.get.return_value = mock_network

        manager._remove_existing_network()

        mock_network.disconnect.assert_any_call(mock_c1, force=True)
        mock_network.disconnect.assert_any_call(mock_c2, force=True)
        mock_network.remove.assert_called_once()

    def test_removes_network_with_no_containers(self):
        """Network with no attached containers is removed directly."""
        manager = self._make_manager()
        mock_network = MagicMock()
        mock_network.containers = []
        manager._client.networks.get.return_value = mock_network

        manager._remove_existing_network()

        mock_network.disconnect.assert_not_called()
        mock_network.remove.assert_called_once()

    def test_handles_not_found_gracefully(self):
        """If the network does not exist, the method returns without error."""
        from docker.errors import NotFound

        manager = self._make_manager()
        manager._client.networks.get.side_effect = NotFound("gone")

        manager._remove_existing_network()

    def test_continues_disconnecting_on_individual_failure(self):
        """If one container fails to disconnect, others are still attempted."""
        manager = self._make_manager()
        mock_c1 = MagicMock(name="bad-container")
        mock_c2 = MagicMock(name="good-container")
        mock_network = MagicMock()
        mock_network.containers = [mock_c1, mock_c2]
        mock_network.disconnect.side_effect = [Exception("failed"), None]
        manager._client.networks.get.return_value = mock_network

        manager._remove_existing_network()

        assert mock_network.disconnect.call_count == 2
        mock_network.remove.assert_called_once()

    def test_reload_failure_still_attempts_remove(self):
        """If network.reload() fails, the method still attempts network.remove()."""
        manager = self._make_manager()
        mock_network = MagicMock()
        mock_network.reload.side_effect = Exception("connection reset")
        mock_network.containers = []
        manager._client.networks.get.return_value = mock_network

        manager._remove_existing_network()

        mock_network.remove.assert_called_once()

    def test_non_iterable_containers_skips_disconnect(self):
        """If network.containers is non-iterable, skip disconnect and still remove."""
        manager = self._make_manager()
        mock_network = MagicMock()
        mock_network.containers = 42
        manager._client.networks.get.return_value = mock_network

        manager._remove_existing_network()

        mock_network.disconnect.assert_not_called()
        mock_network.remove.assert_called_once()

    def test_none_containers_skips_disconnect(self):
        """If network.containers is None, skip disconnect and still remove."""
        manager = self._make_manager()
        mock_network = MagicMock()
        mock_network.containers = None
        manager._client.networks.get.return_value = mock_network

        manager._remove_existing_network()

        mock_network.disconnect.assert_not_called()
        mock_network.remove.assert_called_once()

    def test_active_endpoints_error_raises_with_recovery_steps(self):
        """APIError with 'active endpoints' raises RuntimeError with recovery instructions."""
        from docker.errors import APIError

        manager = self._make_manager()
        mock_network = MagicMock()
        mock_network.containers = []
        mock_network.remove.side_effect = APIError("network has active endpoints")
        manager._client.networks.get.return_value = mock_network

        with pytest.raises(RuntimeError, match="still connected") as exc_info:
            manager._remove_existing_network()

        msg = str(exc_info.value)
        assert "nhx-quickstart-network" in msg
        assert "docker network inspect" in msg
        assert "docker stop" in msg

    def test_other_api_error_is_reraised(self):
        """Non-active-endpoints APIError is logged and re-raised."""
        from docker.errors import APIError

        manager = self._make_manager()
        mock_network = MagicMock()
        mock_network.containers = []
        mock_network.remove.side_effect = APIError("server error")
        manager._client.networks.get.return_value = mock_network

        with pytest.raises(APIError, match="server error"):
            manager._remove_existing_network()


class TestStartCleansUpBeforeCreatingNetwork:
    """Tests that start() removes model deployments before removing the network."""

    def test_start_calls_remove_model_deployments_before_remove_network(self):
        """start() must call _remove_model_deployments before _remove_existing_network."""
        config = MagicMock()
        config.image = "nvcr.io/nvidia/nhx-api:25.01"
        config.container_name = "nhx-quickstart"
        config.network_name = "nhx-quickstart-network"
        config.host_port = 8080
        config.container_port = 8080
        config.container_id = None
        config.save = MagicMock()

        manager = ContainerManager.__new__(ContainerManager)
        manager.config = config

        mock_client = MagicMock()
        mock_network = MagicMock()
        mock_network.name = "nhx-quickstart-network"
        mock_client.networks.create.return_value = mock_network
        mock_container = MagicMock()
        mock_container.id = "abc123"
        mock_client.containers.run.return_value = mock_container
        manager._client = mock_client

        call_order: list[str] = []

        with (
            patch.object(manager, "_create_mounts", return_value=[]),
            patch.object(manager, "_create_environment", return_value={}),
            patch.object(manager, "_remove_existing_container", side_effect=lambda: call_order.append("container")),
            patch.object(
                manager, "_remove_model_deployments", side_effect=lambda **kw: call_order.append("deployments")
            ),
            patch.object(manager, "_remove_existing_network", side_effect=lambda: call_order.append("network")),
        ):
            manager.start(platform_config=MagicMock(), pull=False)

        assert call_order == ["container", "deployments", "network"]


class TestReconstructConfigFromContainer:
    """Tests for ContainerManager.reconstruct_config_from_container."""

    def _make_mock_container(self) -> MagicMock:
        """Return a MagicMock container with realistic inspect data."""
        container = MagicMock()
        container.name = "/nhx-quickstart"
        container.image.tags = ["nvcr.io/nvidia/nemo-microservices/nhx-api:25.01"]
        container.attrs = {
            "Config": {
                "Labels": {
                    "nhx.nvidia.com/host-port": "8080",
                },
            },
            "HostConfig": {
                "PortBindings": {
                    "8080/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8080"}],
                },
            },
            "Mounts": [
                {
                    "Type": "bind",
                    "Source": "/var/run/docker.sock",
                    "Destination": "/var/run/docker.sock",
                },
                {
                    "Type": "bind",
                    "Source": "/home/user/.config/nhx/quickstart/data",
                    "Destination": "/data",
                },
            ],
            "NetworkSettings": {
                "Networks": {
                    "nhx-quickstart-network": {},
                },
            },
        }
        return container

    def test_reconstructs_image_from_container_tags(self):
        container = self._make_mock_container()
        config = ContainerManager.reconstruct_config_from_container(container)
        assert config.image == "nvcr.io/nvidia/nemo-microservices/nhx-api:25.01"

    def test_reconstructs_host_port_from_port_bindings(self):
        container = self._make_mock_container()
        # Use a distinct port so we confirm it's read correctly
        container.attrs["HostConfig"]["PortBindings"] = {
            "8080/tcp": [{"HostIp": "0.0.0.0", "HostPort": "9090"}],
        }
        container.attrs["Config"]["Labels"]["nhx.nvidia.com/host-port"] = "9090"
        config = ContainerManager.reconstruct_config_from_container(container)
        assert config.host_port == 9090

    def test_reconstructs_storage_path_from_data_mount(self):
        container = self._make_mock_container()
        config = ContainerManager.reconstruct_config_from_container(container)
        assert config.storage_path == Path("/home/user/.config/nhx/quickstart")

    def test_reconstructs_docker_socket_from_mount(self):
        container = self._make_mock_container()
        config = ContainerManager.reconstruct_config_from_container(container)
        assert config.docker_socket == Path("/var/run/docker.sock")

    def test_reconstructs_network_name_from_networks(self):
        container = self._make_mock_container()
        config = ContainerManager.reconstruct_config_from_container(container)
        assert config.network_name == "nhx-quickstart-network"


class TestContainerManagerStartLabels:
    """Tests that start() attaches the expected Docker labels to the container."""

    def _make_manager(self, **config_overrides) -> tuple[ContainerManager, MagicMock]:
        """Create a ContainerManager with a mock config and Docker client."""
        config = MagicMock()
        config.image = config_overrides.get("image", "nvcr.io/nvidia/nhx-api:25.01")
        config.container_name = config_overrides.get("container_name", "nhx-quickstart")
        config.network_name = config_overrides.get("network_name", "nhx-quickstart-network")
        config.host_port = config_overrides.get("host_port", 8080)
        config.container_port = config_overrides.get("container_port", 8080)
        config.inference_provider = config_overrides.get("inference_provider", None)
        config.use_gpu = config_overrides.get("use_gpu", False)
        config.auth_enabled = config_overrides.get("auth_enabled", False)
        config.container_id = None
        config.save = MagicMock()

        manager = ContainerManager.__new__(ContainerManager)
        manager.config = config

        mock_client = MagicMock()
        mock_network = MagicMock()
        mock_network.name = "nhx-quickstart-network"
        mock_client.networks.create.return_value = mock_network
        mock_container = MagicMock()
        mock_container.id = "abc123"
        mock_client.containers.run.return_value = mock_container
        manager._client = mock_client

        return manager, mock_client

    def _start_with_mocked_helpers(self, manager: ContainerManager) -> None:
        """Call start() with all internal helpers stubbed out."""
        with (
            patch.object(manager, "_create_mounts", return_value=[]),
            patch.object(manager, "_create_environment", return_value={}),
            patch.object(manager, "_remove_existing_container"),
            patch.object(manager, "_remove_model_deployments"),
            patch.object(manager, "_remove_existing_network"),
        ):
            manager.start(platform_config=MagicMock(), pull=False)

    def test_start_sets_inference_provider_label(self):
        manager, mock_client = self._make_manager(inference_provider="nvidia-build")
        self._start_with_mocked_helpers(manager)
        labels = mock_client.containers.run.call_args.kwargs["labels"]
        assert labels["nhx.nvidia.com/inference-provider"] == "nvidia-build"

    def test_start_sets_use_gpu_label_true_when_use_gpu_enabled(self):
        manager, mock_client = self._make_manager(use_gpu=True)
        self._start_with_mocked_helpers(manager)
        labels = mock_client.containers.run.call_args.kwargs["labels"]
        assert labels["nhx.nvidia.com/use-gpu"] == "true"

    def test_start_sets_auth_enabled_label(self):
        manager, mock_client = self._make_manager(auth_enabled=True)
        self._start_with_mocked_helpers(manager)
        labels = mock_client.containers.run.call_args.kwargs["labels"]
        assert labels["nhx.nvidia.com/auth-enabled"] == "true"

    def test_start_sets_host_port_label(self):
        manager, mock_client = self._make_manager(host_port=9090)
        self._start_with_mocked_helpers(manager)
        labels = mock_client.containers.run.call_args.kwargs["labels"]
        assert labels["nhx.nvidia.com/host-port"] == "9090"


class TestImagePullAuthOverride:
    """Tests for temporary auth during image pull."""

    def test_pull_image_with_progress_uses_auth_override(self):
        """When override auth is provided, it should be sent to docker pull."""
        manager = ContainerManager.__new__(ContainerManager)
        manager.config = MagicMock()
        manager.config.image = "registry.example.com/repo/nhx-api:test"

        api_client = MagicMock()
        api_client.pull.return_value = iter(
            [
                {
                    "status": "Pulling fs layer",
                    "id": "layer-1",
                    "progressDetail": {"current": 1, "total": 10},
                }
            ]
        )
        manager._client = MagicMock()
        manager._client.api = api_client

        with patch.object(ContainerManager, "is_image_available", return_value=False):
            list(
                manager.pull_image_with_progress(
                    auth_override={
                        "registry": "registry.example.com",
                        "username": "test-user",
                        "password": "test-pass",
                    }
                )
            )

        api_client.pull.assert_called_once_with(
            "registry.example.com/repo/nhx-api:test",
            stream=True,
            decode=True,
            auth_config={"username": "test-user", "password": "test-pass"},
        )

    def test_pull_image_with_progress_uses_auth_override_for_two_part_registry_image(self):
        """Two-part registry images should still match the override registry host."""
        manager = ContainerManager.__new__(ContainerManager)
        manager.config = MagicMock()
        manager.config.image = "registry.example.com/nhx-api:test"

        api_client = MagicMock()
        api_client.pull.return_value = iter(
            [
                {
                    "status": "Pulling fs layer",
                    "id": "layer-1",
                    "progressDetail": {"current": 1, "total": 10},
                }
            ]
        )
        manager._client = MagicMock()
        manager._client.api = api_client

        with patch.object(ContainerManager, "is_image_available", return_value=False):
            list(
                manager.pull_image_with_progress(
                    auth_override={
                        "registry": "registry.example.com",
                        "username": "test-user",
                        "password": "test-pass",
                    }
                )
            )

        api_client.pull.assert_called_once_with(
            "registry.example.com/nhx-api:test",
            stream=True,
            decode=True,
            auth_config={"username": "test-user", "password": "test-pass"},
        )
