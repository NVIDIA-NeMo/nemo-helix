# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``NamespacedModel`` backend-schema base.

These pin the behaviour the customization OpenAPI fix relies on: each backend's
models emit a namespace-prefixed schema name at *class-creation time*, so two
backends that each define e.g. ``TrainingSpec`` don't collapse into one component
when merged into the single ``/apis/customization`` FastAPI app.
"""

import pytest
from nhx.customization_common.schema import NamespacedModel
from pydantic import Field, ValidationError


class AutomodelSchema(NamespacedModel):
    __schema_namespace__ = "Automodel"


class UnslothSchema(NamespacedModel):
    __schema_namespace__ = "Unsloth"


# Two backends each define a leaf literally named ``TrainingSpec`` — exactly the
# collision the metaclass exists to prevent. Capture each binding under an alias
# before the next ``class TrainingSpec`` statement shadows the module name.
class TrainingSpec(AutomodelSchema):
    lr: float = 1e-4


AutomodelTrainingSpec = TrainingSpec


class TrainingSpec(UnslothSchema):  # noqa: F811 — intentional same-name redefinition
    steps: int = 100


UnslothTrainingSpec = TrainingSpec


class AutomodelJobInput(AutomodelSchema):
    training: AutomodelTrainingSpec = Field(default_factory=AutomodelTrainingSpec)


def test_leaf_models_are_prefixed_at_class_creation():
    assert AutomodelTrainingSpec.__name__ == "AutomodelTrainingSpec"
    assert UnslothTrainingSpec.__name__ == "UnslothTrainingSpec"


def test_same_leaf_name_across_backends_emits_distinct_component_names():
    a_title = AutomodelTrainingSpec.model_json_schema()["title"]
    u_title = UnslothTrainingSpec.model_json_schema()["title"]
    assert a_title == "AutomodelTrainingSpec"
    assert u_title == "UnslothTrainingSpec"
    # The two normalize to different bare names, so the platform normalizer won't
    # collapse them into one component.
    assert a_title != u_title


def test_nested_ref_points_at_the_owning_backends_schema():
    schema = AutomodelJobInput.model_json_schema()
    assert "AutomodelTrainingSpec" in schema["$defs"]
    assert schema["properties"]["training"]["$ref"].endswith("/AutomodelTrainingSpec")


def test_top_level_name_already_prefixed_is_not_double_prefixed():
    assert AutomodelJobInput.__name__ == "AutomodelJobInput"
    assert AutomodelJobInput.model_json_schema()["title"] == "AutomodelJobInput"


def test_backend_base_class_itself_is_not_prefixed():
    # The intermediate per-backend base keeps its own name (it is never emitted
    # as a request/response schema, but must not be mangled either).
    assert AutomodelSchema.__name__ == "AutomodelSchema"
    assert UnslothSchema.__name__ == "UnslothSchema"


def test_extra_forbid_is_inherited():
    assert AutomodelJobInput.model_config.get("extra") == "forbid"
    assert AutomodelJobInput.model_json_schema()["additionalProperties"] is False
    with pytest.raises(ValidationError):
        # Extra field rejected (validated via dict so the extra key isn't a
        # static type error); training defaults, so ``bogus`` is the only fault.
        AutomodelJobInput.model_validate({"training": {}, "bogus": 1})


def test_schema_namespace_is_not_a_field():
    assert "__schema_namespace__" not in AutomodelJobInput.model_fields
    assert AutomodelSchema.__schema_namespace__ == "Automodel"


def test_plain_namespaced_model_without_namespace_is_unprefixed():
    class Bare(NamespacedModel):
        x: int = 0

    assert Bare.__name__ == "Bare"
    assert Bare.model_json_schema()["title"] == "Bare"


def test_subclass_with_own_model_config_still_inherits_extra_forbid():
    """A subclass that declares its OWN ``model_config`` must still inherit
    ``extra='forbid'`` from the base.

    ``RlJobInput``, ``_TrainingBase`` and ``RlJobOutput`` each set
    ``ConfigDict(protected_namespaces=())`` and rely on pydantic *merging* (not
    replacing) the base config to keep rejecting unknown fields. If that ever
    regressed, ``additionalProperties: false`` would silently vanish from those
    request bodies and this suite would still pass without this guard.
    """
    from pydantic import ConfigDict

    class WithOwnConfig(AutomodelSchema):
        model_config = ConfigDict(protected_namespaces=())

        value: int = 0

    # Subclass's own key applied *and* the base's ``extra='forbid'`` preserved.
    assert WithOwnConfig.model_config.get("protected_namespaces") == ()
    assert WithOwnConfig.model_config.get("extra") == "forbid"
    assert WithOwnConfig.model_json_schema()["additionalProperties"] is False
    with pytest.raises(ValidationError):
        WithOwnConfig.model_validate({"value": 1, "bogus": 2})
