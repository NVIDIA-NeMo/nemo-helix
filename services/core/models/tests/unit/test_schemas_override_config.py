# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ContainerExecutorConfig override_config field validation.

Tests ensure that the override_config field properly validates against NIMService Spec schema
and can be serialized/deserialized correctly for Kubernetes resource application.
"""

import json
from datetime import datetime

import pytest
from nimservice_spec_types import Spec
from nhx.core.models.schemas import (
    ContainerExecutorConfig,
    CreateModelDeploymentConfigRequest,
    Engine,
    ModelDeploymentConfig,
    ModelDeploymentConfigModelSpec,
)
from pydantic import ValidationError

# ============================================================================
# Basic ContainerExecutorConfig override_config Tests
# ============================================================================


def test_nim_deployment_without_override_config():
    """Test that ContainerExecutorConfig works without override_config (backwards compatibility)."""
    executor_config = ContainerExecutorConfig(
        gpu=1,
        image_name="nvcr.io/nvidia/nim/llm",
        image_tag="latest",
    )

    assert executor_config.gpu == 1
    assert executor_config.override_config is None


def test_nim_deployment_with_empty_override_config():
    """Test that ContainerExecutorConfig accepts an empty dict for override_config."""
    executor_config = ContainerExecutorConfig(
        gpu=1,
        override_config={},
    )

    assert executor_config.gpu == 1
    assert executor_config.override_config == {}


def test_nim_deployment_with_valid_override_config_basic():
    """Test ContainerExecutorConfig with basic valid override_config matching Spec structure."""
    # Create a basic override config with only required Spec fields
    override_config = {
        "authSecret": "my-ngc-secret",
        "image": {
            "repository": "nvcr.io/nvidia/nim/custom-llm",
            "tag": "1.0.0",
        },
    }

    executor_config = ContainerExecutorConfig(
        gpu=1,
        override_config=override_config,
    )

    assert executor_config.override_config is not None
    assert executor_config.override_config["authSecret"] == "my-ngc-secret"
    assert executor_config.override_config["image"]["repository"] == "nvcr.io/nvidia/nim/custom-llm"


def test_nim_deployment_with_valid_override_config_complex():
    """Test ContainerExecutorConfig with complex valid override_config including many Spec fields."""
    override_config = {
        "authSecret": "my-ngc-secret",
        "image": {
            "repository": "nvcr.io/nvidia/nim/llm",
            "tag": "latest",
            "pullPolicy": "Always",
            "pullSecrets": ["my-pull-secret"],
        },
        "replicas": 3,
        "env": [
            {"name": "NIM_CACHE_PATH", "value": "/model-store"},
            {"name": "LOG_LEVEL", "value": "DEBUG"},
        ],
        "resources": {
            "limits": {"nvidia.com/gpu": "2", "memory": "16Gi"},
            "requests": {"nvidia.com/gpu": "2", "memory": "16Gi"},
        },
        "nodeSelector": {"gpu-type": "a100"},
        "tolerations": [
            {
                "key": "nvidia.com/gpu",
                "operator": "Equal",
                "value": "true",
                "effect": "NoSchedule",
            }
        ],
        "annotations": {"prometheus.io/scrape": "true"},
        "labels": {"app": "nim-service", "environment": "production"},
    }

    executor_config = ContainerExecutorConfig(
        gpu=2,
        override_config=override_config,
    )

    assert executor_config.override_config is not None
    assert executor_config.override_config["replicas"] == 3
    assert executor_config.override_config["authSecret"] == "my-ngc-secret"
    assert len(executor_config.override_config["env"]) == 2
    assert executor_config.override_config["nodeSelector"]["gpu-type"] == "a100"


def test_nim_deployment_override_config_with_storage():
    """Test override_config with storage configuration."""
    override_config = {
        "authSecret": "my-ngc-secret",
        "image": {"repository": "nvcr.io/nvidia/nim/llm", "tag": "latest"},
        "storage": {
            "pvc": {
                "create": True,
                "name": "nim-cache-pvc",
                "size": "100Gi",
                "storageClass": "fast-ssd",
                "volumeAccessMode": "ReadWriteOnce",
            }
        },
    }

    executor_config = ContainerExecutorConfig(
        gpu=1,
        override_config=override_config,
    )

    assert executor_config.override_config["storage"]["pvc"]["size"] == "100Gi"
    assert executor_config.override_config["storage"]["pvc"]["storageClass"] == "fast-ssd"


def test_nim_deployment_override_config_with_nimcache():
    """Test override_config with NIMCache storage configuration."""
    override_config = {
        "authSecret": "my-ngc-secret",
        "image": {"repository": "nvcr.io/nvidia/nim/llm", "tag": "latest"},
        "storage": {
            "nimCache": {
                "name": "llama-3-8b-cache",
                "profile": "tp2-gpu-a100",
            }
        },
    }

    executor_config = ContainerExecutorConfig(
        gpu=2,
        override_config=override_config,
    )

    assert executor_config.override_config["storage"]["nimCache"]["name"] == "llama-3-8b-cache"
    assert executor_config.override_config["storage"]["nimCache"]["profile"] == "tp2-gpu-a100"


def test_nim_deployment_override_config_with_probes():
    """Test override_config with health probe configurations."""
    override_config = {
        "authSecret": "my-ngc-secret",
        "image": {"repository": "nvcr.io/nvidia/nim/llm", "tag": "latest"},
        "livenessProbe": {
            "enabled": True,
            "probe": {
                "httpGet": {"path": "/health", "port": 8000},
                "initialDelaySeconds": 30,
                "periodSeconds": 10,
            },
        },
        "readinessProbe": {
            "enabled": True,
            "probe": {
                "httpGet": {"path": "/ready", "port": 8000},
                "initialDelaySeconds": 10,
                "periodSeconds": 5,
            },
        },
    }

    executor_config = ContainerExecutorConfig(
        gpu=1,
        override_config=override_config,
    )

    assert executor_config.override_config["livenessProbe"]["enabled"] is True
    assert executor_config.override_config["readinessProbe"]["probe"]["httpGet"]["path"] == "/ready"


def test_nim_deployment_override_config_with_autoscaling():
    """Test override_config with HPA autoscaling configuration."""
    override_config = {
        "authSecret": "my-ngc-secret",
        "image": {"repository": "nvcr.io/nvidia/nim/llm", "tag": "latest"},
        "scale": {
            "enabled": True,
            "hpa": {
                "minReplicas": 1,
                "maxReplicas": 10,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {"type": "Utilization", "averageUtilization": 80},
                        },
                    }
                ],
            },
        },
    }

    executor_config = ContainerExecutorConfig(
        gpu=1,
        override_config=override_config,
    )

    assert executor_config.override_config["scale"]["enabled"] is True
    assert executor_config.override_config["scale"]["hpa"]["maxReplicas"] == 10


def test_nim_deployment_override_config_with_service_exposure():
    """Test override_config with service and ingress exposure."""
    override_config = {
        "authSecret": "my-ngc-secret",
        "image": {"repository": "nvcr.io/nvidia/nim/llm", "tag": "latest"},
        "expose": {
            "service": {
                "type": "ClusterIP",
                "port": 8000,
                "annotations": {"service.beta.kubernetes.io/aws-load-balancer-type": "nlb"},
            },
            "ingress": {
                "enabled": True,
                "annotations": {"kubernetes.io/ingress.class": "nginx"},
                "spec": {
                    "rules": [
                        {
                            "host": "nim.example.com",
                            "http": {
                                "paths": [
                                    {
                                        "path": "/",
                                        "pathType": "Prefix",
                                        "backend": {
                                            "service": {
                                                "name": "nim-service",
                                                "port": {"number": 8000},
                                            }
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                },
            },
        },
    }

    executor_config = ContainerExecutorConfig(
        gpu=1,
        override_config=override_config,
    )

    assert executor_config.override_config["expose"]["service"]["type"] == "ClusterIP"
    assert executor_config.override_config["expose"]["ingress"]["enabled"] is True


def test_nim_deployment_override_config_serialization():
    """Test that override_config is properly serialized to dict."""
    override_config = {
        "authSecret": "my-secret",
        "image": {"repository": "nvcr.io/test", "tag": "v1"},
        "replicas": 2,
    }

    executor_config = ContainerExecutorConfig(
        gpu=1,
        override_config=override_config,
    )

    # Serialize to dict
    serialized = executor_config.model_dump()
    assert "override_config" in serialized
    assert serialized["override_config"]["authSecret"] == "my-secret"
    assert serialized["override_config"]["replicas"] == 2


def test_nim_deployment_override_config_json_serialization():
    """Test that override_config can be serialized to JSON and back."""
    override_config = {
        "authSecret": "my-secret",
        "image": {"repository": "nvcr.io/test", "tag": "v1"},
        "env": [{"name": "TEST", "value": "value"}],
    }

    executor_config = ContainerExecutorConfig(
        gpu=1,
        override_config=override_config,
    )

    # Serialize to JSON
    json_str = executor_config.model_dump_json()
    assert "override_config" in json_str
    assert "authSecret" in json_str

    # Deserialize from JSON
    executor_config_from_json = ContainerExecutorConfig.model_validate_json(json_str)
    assert executor_config_from_json.override_config["authSecret"] == "my-secret"
    assert executor_config_from_json.override_config["env"][0]["name"] == "TEST"


def test_create_deployment_config_request_with_override_config():
    """Test CreateModelDeploymentConfigRequest with override_config."""
    override_config = {
        "authSecret": "my-secret",
        "image": {"repository": "nvcr.io/test", "tag": "latest"},
    }

    request = CreateModelDeploymentConfigRequest(
        name="test-config",
        engine=Engine.NIM,
        model_spec=ModelDeploymentConfigModelSpec(),
        executor_config=ContainerExecutorConfig(gpu=1, override_config=override_config),
    )

    assert request.executor_config.override_config is not None
    assert request.executor_config.override_config["authSecret"] == "my-secret"


def test_model_deployment_config_with_override_config():
    """Test ModelDeploymentConfig with override_config in executor_config."""
    override_config = {
        "authSecret": "my-secret",
        "image": {"repository": "nvcr.io/test", "tag": "latest"},
        "replicas": 3,
    }

    config = ModelDeploymentConfig(
        id="config-1",
        name="test-config",
        workspace="default",
        entity_version=1,
        engine=Engine.NIM,
        model_spec=ModelDeploymentConfigModelSpec(),
        executor_config=ContainerExecutorConfig(gpu=1, override_config=override_config),
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    assert config.executor_config.override_config is not None
    assert config.executor_config.override_config["replicas"] == 3


def test_override_config_does_not_override_existing_fields():
    """Test that override_config is stored separately and doesn't affect other ContainerExecutorConfig fields."""
    override_config = {
        "authSecret": "override-secret",
        "image": {"repository": "nvcr.io/override", "tag": "override-tag"},
    }

    executor_config = ContainerExecutorConfig(
        gpu=2,
        image_name="nvcr.io/nvidia/nim/llm",
        image_tag="latest",
        override_config=override_config,
    )

    # Verify that original fields are preserved
    assert executor_config.gpu == 2
    assert executor_config.image_name == "nvcr.io/nvidia/nim/llm"
    assert executor_config.image_tag == "latest"

    # Verify override_config is stored separately
    assert executor_config.override_config["authSecret"] == "override-secret"
    assert executor_config.override_config["image"]["repository"] == "nvcr.io/override"


# ============================================================================
# Spec Validation Tests
# ============================================================================


def test_override_config_validates_as_partial_spec():
    """Test that override_config can contain valid Spec fields."""
    # Create a valid Spec-compatible dict
    spec_dict = {
        "authSecret": "my-ngc-secret",
        "image": {
            "repository": "nvcr.io/nvidia/nim/llm",
            "tag": "latest",
        },
        "replicas": 1,
    }

    # This should work - Dict[str, Any] accepts any structure
    executor_config = ContainerExecutorConfig(gpu=1, override_config=spec_dict)
    assert executor_config.override_config is not None

    # Verify that the dict structure matches what Spec expects (manual validation)
    # Note: We're NOT validating as Spec here, just ensuring the structure is compatible
    assert "authSecret" in executor_config.override_config
    assert "image" in executor_config.override_config
    assert "repository" in executor_config.override_config["image"]
    assert "tag" in executor_config.override_config["image"]


def test_validate_override_config_against_spec_manually():
    """Test manual validation of override_config against Spec model.

    This demonstrates how to validate override_config as a Spec object
    when needed (e.g., before applying to k8s).
    """
    override_config = {
        "authSecret": "my-ngc-secret",
        "image": {
            "repository": "nvcr.io/nvidia/nim/llm",
            "tag": "latest",
            "pullPolicy": "IfNotPresent",
        },
        "replicas": 2,
        "resources": {
            "limits": {"nvidia.com/gpu": 1, "memory": "16Gi"},
            "requests": {"nvidia.com/gpu": 1, "memory": "16Gi"},
        },
        "nodeSelector": {"gpu-type": "a100"},
    }

    executor_config = ContainerExecutorConfig(gpu=1, override_config=override_config)

    # Manual validation: try to construct a Spec object from override_config
    # This would be done by the NIMService compiler before k8s application
    try:
        spec = Spec(**executor_config.override_config)
        assert spec.authSecret == "my-ngc-secret"
        assert spec.replicas == 2
        assert spec.image.repository == "nvcr.io/nvidia/nim/llm"
        assert spec.nodeSelector["gpu-type"] == "a100"
        # Resources use RootModel, access the .root attribute
        assert spec.resources.limits["nvidia.com/gpu"].root == 1
    except ValidationError as e:
        pytest.fail(f"override_config failed Spec validation: {e}")


def test_invalid_spec_structure_accepted_by_dict():
    """Test that invalid Spec structures are accepted by Dict[str, Any] (intentional).

    This is expected behavior - the Dict[str, Any] type doesn't validate structure.
    Validation happens later when compiling to NIMService.
    """
    # This has invalid structure for Spec (missing required fields)
    invalid_override_config = {
        "some_random_field": "value",
        "nested": {"invalid": "structure"},
    }

    # This should NOT raise an error - Dict[str, Any] accepts anything
    executor_config = ContainerExecutorConfig(gpu=1, override_config=invalid_override_config)
    assert executor_config.override_config is not None

    # But when we try to validate as Spec, it should fail
    with pytest.raises(ValidationError):
        Spec(**invalid_override_config)


def test_spec_validation_with_all_optional_fields():
    """Test that Spec can be validated with many optional fields from override_config."""
    override_config = {
        "authSecret": "my-secret",
        "image": {
            "repository": "nvcr.io/nvidia/nim/llm",
            "tag": "latest",
            "pullPolicy": "Always",
            "pullSecrets": ["my-pull-secret"],
        },
        "replicas": 3,
        "env": [
            {"name": "VAR1", "value": "value1"},
            {"name": "VAR2", "value": "value2"},
        ],
        "resources": {
            "limits": {"nvidia.com/gpu": 2, "memory": "16Gi"},
            "requests": {"nvidia.com/gpu": 2, "memory": "16Gi"},
        },
        "storage": {
            "pvc": {
                "create": True,
                "name": "cache-pvc",
                "size": "100Gi",
            }
        },
        "nodeSelector": {"node-type": "gpu"},
        "tolerations": [
            {
                "key": "nvidia.com/gpu",
                "operator": "Exists",
                "effect": "NoSchedule",
            }
        ],
    }

    executor_config = ContainerExecutorConfig(gpu=2, override_config=override_config)

    # Validate as Spec
    spec = Spec(**executor_config.override_config)
    assert spec.authSecret == "my-secret"
    assert spec.replicas == 3
    assert len(spec.env) == 2
    assert spec.storage.pvc.size == "100Gi"
    assert spec.nodeSelector["node-type"] == "gpu"
    # Resources use RootModel, access the .root attribute
    assert spec.resources.limits["nvidia.com/gpu"].root == 2


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


def test_override_config_with_none_value():
    """Test that override_config can be explicitly set to None."""
    executor_config = ContainerExecutorConfig(gpu=1, override_config=None)
    assert executor_config.override_config is None


def test_override_config_with_nested_none_values():
    """Test override_config with None values in nested structures."""
    override_config = {
        "authSecret": "my-secret",
        "image": {"repository": "nvcr.io/test", "tag": "latest"},
        "description": None,  # Optional field explicitly set to None
        "annotations": None,
    }

    executor_config = ContainerExecutorConfig(gpu=1, override_config=override_config)
    assert executor_config.override_config["description"] is None


def test_override_config_deeply_nested_structure():
    """Test override_config with deeply nested structures."""
    override_config = {
        "authSecret": "my-secret",
        "image": {"repository": "nvcr.io/test", "tag": "latest"},
        "expose": {
            "ingress": {
                "spec": {
                    "rules": [
                        {
                            "host": "test.com",
                            "http": {
                                "paths": [
                                    {
                                        "path": "/api",
                                        "pathType": "Prefix",
                                        "backend": {
                                            "service": {
                                                "name": "test-svc",
                                                "port": {"number": 8000},
                                            }
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                }
            }
        },
    }

    executor_config = ContainerExecutorConfig(gpu=1, override_config=override_config)
    assert executor_config.override_config["expose"]["ingress"]["spec"]["rules"][0]["host"] == "test.com"


def test_override_config_with_numeric_string_values():
    """Test that numeric strings in override_config are preserved."""
    override_config = {
        "authSecret": "my-secret",
        "image": {"repository": "nvcr.io/test", "tag": "latest"},
        "resources": {
            "limits": {
                "nvidia.com/gpu": "2",  # String, not int
                "memory": "16Gi",  # Kubernetes quantity as string
            }
        },
    }

    executor_config = ContainerExecutorConfig(gpu=1, override_config=override_config)
    assert isinstance(executor_config.override_config["resources"]["limits"]["nvidia.com/gpu"], str)
    assert executor_config.override_config["resources"]["limits"]["nvidia.com/gpu"] == "2"


def test_override_config_serialization_preserves_types():
    """Test that serialization/deserialization preserves data types in override_config."""
    override_config = {
        "authSecret": "my-secret",
        "image": {"repository": "nvcr.io/test", "tag": "latest"},
        "replicas": 3,  # int
        "userID": 1000,  # int
        "groupID": 1000,  # int
        "resources": {
            "limits": {"memory": "16Gi"},  # string
        },
        "annotations": {"key": "value"},  # dict
        "env": [{"name": "VAR", "value": "val"}],  # list of dicts
    }

    executor_config = ContainerExecutorConfig(gpu=1, override_config=override_config)

    # Serialize and deserialize
    json_str = executor_config.model_dump_json()
    restored = ContainerExecutorConfig.model_validate_json(json_str)

    # Verify types are preserved
    assert isinstance(restored.override_config["replicas"], int)
    assert isinstance(restored.override_config["userID"], int)
    assert isinstance(restored.override_config["resources"]["limits"]["memory"], str)
    assert isinstance(restored.override_config["env"], list)


# ============================================================================
# Round-trip Serialization Tests
# ============================================================================


def test_override_config_roundtrip_serialization_with_spec_validation():
    """Test that override_config survives dict -> Spec validation -> dict round-trip.

    This is the critical test for the NIMService compiler flow:
    1. Start with dict (override_config)
    2. Validate as Spec (ensure it's valid)
    3. Serialize back to dict (for K8s application)
    4. Verify output matches input
    """
    original_config = {
        "authSecret": "my-ngc-secret",
        "image": {
            "repository": "nvcr.io/nvidia/nim/llm",
            "tag": "latest",
            "pullPolicy": "IfNotPresent",
        },
        "replicas": 2,
        "resources": {
            "limits": {"nvidia.com/gpu": 1, "memory": "16Gi"},
            "requests": {"nvidia.com/gpu": 1, "memory": "8Gi"},
        },
        "env": [
            {"name": "MODEL_NAME", "value": "llama-3-8b"},
            {"name": "MAX_BATCH_SIZE", "value": "32"},
        ],
        "nodeSelector": {"gpu-type": "a100"},
    }

    # Step 1: Create ContainerExecutorConfig with override_config
    executor_config = ContainerExecutorConfig(gpu=1, override_config=original_config)

    # Step 2: Validate as Spec (this is what NIMService compiler would do)
    spec = Spec(**executor_config.override_config)

    # Step 3: Serialize back to dict (for K8s application)
    serialized_spec = spec.model_dump(exclude_none=True)

    # Step 4: Verify structure matches original
    assert serialized_spec["authSecret"] == original_config["authSecret"]
    assert serialized_spec["image"] == original_config["image"]
    assert serialized_spec["replicas"] == original_config["replicas"]

    # Critical: Resources should be plain dict, not RootModel objects
    assert serialized_spec["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert serialized_spec["resources"]["limits"]["memory"] == "16Gi"
    assert serialized_spec["resources"]["requests"]["nvidia.com/gpu"] == 1
    assert serialized_spec["resources"]["requests"]["memory"] == "8Gi"

    # Verify it's a plain int, not a RootModel
    assert isinstance(serialized_spec["resources"]["limits"]["nvidia.com/gpu"], int)

    assert serialized_spec["env"] == original_config["env"]
    assert serialized_spec["nodeSelector"] == original_config["nodeSelector"]


def test_override_config_roundtrip_with_string_gpu_values():
    """Test round-trip with GPU counts as strings (Kubernetes quantity format)."""
    original_config = {
        "authSecret": "my-secret",
        "image": {"repository": "nvcr.io/test", "tag": "v1"},
        "resources": {
            "limits": {"nvidia.com/gpu": "2", "memory": "16Gi"},  # String GPU count
            "requests": {"nvidia.com/gpu": "2", "memory": "16Gi"},
        },
    }

    # Validate as Spec
    spec = Spec(**original_config)

    # Serialize back to dict
    serialized = spec.model_dump(exclude_none=True)

    # Verify GPU count is preserved as string
    assert serialized["resources"]["limits"]["nvidia.com/gpu"] == "2"
    assert isinstance(serialized["resources"]["limits"]["nvidia.com/gpu"], str)
    assert serialized["resources"]["requests"]["nvidia.com/gpu"] == "2"


def test_override_config_roundtrip_with_mixed_resource_types():
    """Test round-trip with mixed int and string resource values."""
    original_config = {
        "authSecret": "my-secret",
        "image": {"repository": "nvcr.io/test", "tag": "v1"},
        "resources": {
            "limits": {
                "nvidia.com/gpu": 4,  # int
                "memory": "32Gi",  # string with unit
                "cpu": "8000m",  # string with milli unit
            },
            "requests": {
                "nvidia.com/gpu": 4,  # int
                "memory": "16Gi",  # string
                "cpu": 4,  # int (cores)
            },
        },
    }

    # Validate as Spec
    spec = Spec(**original_config)

    # Serialize back to dict
    serialized = spec.model_dump(exclude_none=True)

    # Verify all resource types are preserved correctly
    assert serialized["resources"]["limits"]["nvidia.com/gpu"] == 4
    assert isinstance(serialized["resources"]["limits"]["nvidia.com/gpu"], int)

    assert serialized["resources"]["limits"]["memory"] == "32Gi"
    assert isinstance(serialized["resources"]["limits"]["memory"], str)

    assert serialized["resources"]["limits"]["cpu"] == "8000m"
    assert isinstance(serialized["resources"]["limits"]["cpu"], str)

    assert serialized["resources"]["requests"]["cpu"] == 4
    assert isinstance(serialized["resources"]["requests"]["cpu"], int)


def test_override_config_json_roundtrip_through_spec():
    """Test JSON serialization round-trip through Spec validation."""
    original_config = {
        "authSecret": "my-secret",
        "image": {"repository": "nvcr.io/test", "tag": "v1"},
        "replicas": 3,
        "resources": {
            "limits": {"nvidia.com/gpu": 2, "memory": "16Gi"},
        },
    }

    # Validate as Spec
    spec = Spec(**original_config)

    # Serialize to JSON (as would be sent to K8s API)
    json_output = spec.model_dump_json(exclude_none=True)

    # Verify JSON contains correct structure
    parsed = json.loads(json_output)

    assert parsed["authSecret"] == "my-secret"
    assert parsed["replicas"] == 3
    assert parsed["resources"]["limits"]["nvidia.com/gpu"] == 2
    assert parsed["resources"]["limits"]["memory"] == "16Gi"


def test_nim_deployment_to_nimservice_flow():
    """Test the complete flow: ContainerExecutorConfig -> override_config -> Spec -> K8s dict.

    This simulates what the NIMService compiler will do.
    """
    # Step 1: User creates ModelDeploymentConfig with override_config
    override_config = {
        "authSecret": "ngc-secret",
        "image": {
            "repository": "nvcr.io/nvidia/nim/llama-3-8b",
            "tag": "1.0.0",
        },
        "replicas": 2,
        "resources": {
            "limits": {"nvidia.com/gpu": 1, "memory": "16Gi"},
            "requests": {"nvidia.com/gpu": 1, "memory": "16Gi"},
        },
        "storage": {
            "pvc": {
                "create": True,
                "name": "model-cache",
                "size": "100Gi",
                "storageClass": "fast-ssd",
            }
        },
    }

    config = ModelDeploymentConfig(
        id="config-2",
        name="llama-deployment",
        workspace="production",
        entity_version=1,
        engine=Engine.NIM,
        model_spec=ModelDeploymentConfigModelSpec(),
        executor_config=ContainerExecutorConfig(gpu=1, override_config=override_config),
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    # Step 2: NIMService compiler extracts override_config and validates as Spec
    spec = Spec(**config.executor_config.override_config)

    # Step 3: Serialize to dict for K8s application
    k8s_spec = spec.model_dump(exclude_none=True)

    # Step 4: Verify the K8s spec is correct
    assert k8s_spec["authSecret"] == "ngc-secret"
    assert k8s_spec["replicas"] == 2
    assert k8s_spec["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert k8s_spec["storage"]["pvc"]["size"] == "100Gi"

    # Verify it's plain Python types, not Pydantic models
    assert isinstance(k8s_spec, dict)
    assert isinstance(k8s_spec["resources"]["limits"]["nvidia.com/gpu"], int)
    assert isinstance(k8s_spec["resources"]["limits"]["memory"], str)


# ============================================================================
# Integration Tests
# ============================================================================


def test_create_request_to_config_with_override_config():
    """Test full flow from CreateRequest to ModelDeploymentConfig with override_config."""
    override_config = {
        "authSecret": "my-secret",
        "image": {"repository": "nvcr.io/test", "tag": "v1"},
        "replicas": 2,
    }

    # Create request
    create_request = CreateModelDeploymentConfigRequest(
        name="test-config",
        engine=Engine.NIM,
        model_spec=ModelDeploymentConfigModelSpec(),
        executor_config=ContainerExecutorConfig(
            gpu=1,
            override_config=override_config,
        ),
    )

    # Convert to ModelDeploymentConfig (simulating service layer)
    config = ModelDeploymentConfig(
        id="config-3",
        name=create_request.name,
        workspace="default",
        entity_version=1,
        engine=create_request.engine,
        model_spec=create_request.model_spec,
        executor_config=create_request.executor_config,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    # Verify override_config is preserved
    assert config.executor_config.override_config is not None
    assert config.executor_config.override_config["replicas"] == 2
    assert config.executor_config.override_config["authSecret"] == "my-secret"


def test_config_serialization_roundtrip_with_override_config():
    """Test that ModelDeploymentConfig with override_config survives serialization roundtrip."""
    override_config = {
        "authSecret": "my-secret",
        "image": {"repository": "nvcr.io/test", "tag": "v1"},
        "env": [{"name": "TEST", "value": "value"}],
        "resources": {"limits": {"nvidia.com/gpu": "1"}},
    }

    config = ModelDeploymentConfig(
        id="config-1",
        name="test-config",
        workspace="default",
        entity_version=1,
        engine=Engine.NIM,
        model_spec=ModelDeploymentConfigModelSpec(),
        executor_config=ContainerExecutorConfig(gpu=1, override_config=override_config),
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    # Serialize to JSON
    json_str = config.model_dump_json()

    # Deserialize back
    restored_config = ModelDeploymentConfig.model_validate_json(json_str)

    # Verify override_config is intact
    assert restored_config.executor_config.override_config is not None
    assert restored_config.executor_config.override_config["authSecret"] == "my-secret"
    assert len(restored_config.executor_config.override_config["env"]) == 1
    assert restored_config.executor_config.override_config["env"][0]["name"] == "TEST"
