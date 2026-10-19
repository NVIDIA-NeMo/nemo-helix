# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tweak_spec and related OpenAPI spec post-processing."""

import pytest
from nhx.common.api.utils import normalize_schema_name, tweak_spec

REF = "#/components/schemas/"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("FooBar", "FooBar"),
        ("Foo-Input", "FooInput"),
        ("Foo-Output", "FooOutput"),
        ("Page_Job_Filter_", "JobsPage"),
        ("Page_EvaluationConfig__EvaluationConfigFilter_", "EvaluationConfigsPage"),
        ("nemo__api__schemas__ModelInput", "ModelInput"),
        ("nemo__api__Foo-Input", "FooInput"),
        ("InputConfig", "InputConfig"),
        ("OutputConfig", "OutputConfig"),
    ],
)
def test_normalize_schema_name(raw, expected):
    assert normalize_schema_name(raw) == expected


def test_tweak_spec_full_pipeline():
    """Tests the full tweak_spec pipeline using a fully self-contained input/output pair."""
    input_spec = {
        "components": {
            "schemas": {
                "Foo-Input": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
                "Bar-Output": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                },
                "Page_Widget__WidgetFilter_": {
                    "type": "object",
                    "title": "Page_Widget__WidgetFilter_",
                    "properties": {"items": {"type": "array"}},
                },
                "nemo__api__schemas__Baz": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                },
                "Collision": {
                    "type": "object",
                    "properties": {"tag": {"type": "string"}},
                },
                "nemo__api__Collision": {
                    "type": "object",
                    "properties": {"tag": {"type": "string"}},
                },
                "nemo__evaluator__entities__AlphaMetric": {
                    "type": "object",
                    "properties": {"score": {"type": "number"}},
                },
                "nemo__evaluator__entities__BetaMetric": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                },
                "Metric": {
                    "type": "object",
                    "properties": {"score": {"type": "number"}},
                },
                "Entity": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "updated_at": {"type": "string"},
                    },
                },
                "WrapperInput": {
                    "type": "object",
                    "properties": {
                        "item": {"$ref": REF + "Entity"},
                        "items": {"type": "array", "items": {"$ref": REF + "Entity"}},
                    },
                },
                "Clean": {
                    "type": "object",
                    "title": "wrong_title",
                    "properties": {
                        "required_field": {"type": "string"},
                        "optional_field": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "null_first_field": {"anyOf": [{"type": "null"}, {"type": "integer"}]},
                    },
                },
            }
        },
        "paths": {
            "/create": {
                "post": {"requestBody": {"content": {"application/json": {"schema": {"$ref": REF + "Foo-Input"}}}}}
            },
            "/read": {
                "get": {
                    "responses": {"200": {"content": {"application/json": {"schema": {"$ref": REF + "Bar-Output"}}}}}
                }
            },
            "/list": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {"application/json": {"schema": {"$ref": REF + "Page_Widget__WidgetFilter_"}}}
                        }
                    }
                }
            },
            "/lookup": {
                "post": {
                    "requestBody": {
                        "content": {"application/json": {"schema": {"$ref": REF + "nemo__api__schemas__Baz"}}}
                    }
                }
            },
            "/eval": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "discriminator": {
                                        "propertyName": "type",
                                        "mapping": {"m": REF + "Metric-Input"},
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/ref-field": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "anyOf": [
                                        {"type": "string"},
                                        {"$ref": REF + "Foo-Input"},
                                    ]
                                }
                            }
                        }
                    }
                }
            },
            "/evaluate": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "discriminator": {
                                        "propertyName": "type",
                                        "mapping": {
                                            "alpha": REF + "nemo__evaluator__entities__AlphaMetric",
                                            "beta": REF + "nemo__evaluator__entities__BetaMetric",
                                        },
                                    }
                                }
                            }
                        }
                    }
                }
            },
        },
    }

    expected = {
        "components": {
            "schemas": {
                "AlphaMetric": {
                    "type": "object",
                    "title": "AlphaMetric",
                    "properties": {"score": {"type": "number"}},
                },
                "BarOutput": {
                    "type": "object",
                    "title": "BarOutput",
                    "properties": {"value": {"type": "integer"}},
                },
                "Baz": {
                    "type": "object",
                    "title": "Baz",
                    "properties": {"id": {"type": "string"}},
                },
                "BetaMetric": {
                    "type": "object",
                    "title": "BetaMetric",
                    "properties": {"value": {"type": "integer"}},
                },
                "Clean": {
                    "type": "object",
                    "title": "Clean",
                    "properties": {
                        "required_field": {"type": "string"},
                        "optional_field": {"type": "string"},
                        "null_first_field": {"type": "integer"},
                    },
                },
                "Collision": {
                    "type": "object",
                    "title": "Collision",
                    "properties": {"tag": {"type": "string"}},
                },
                "Entity": {
                    "type": "object",
                    "title": "Entity",
                    "properties": {
                        "name": {"type": "string"},
                        "updated_at": {"type": "string"},
                    },
                },
                "EntityInput": {
                    "type": "object",
                    "title": "EntityInput",
                    "properties": {
                        "name": {"type": "string"},
                        "updated_at": {"type": "string"},
                    },
                },
                "FooInput": {
                    "type": "object",
                    "title": "FooInput",
                    "properties": {"name": {"type": "string"}},
                },
                "Metric": {
                    "type": "object",
                    "title": "Metric",
                    "properties": {"score": {"type": "number"}},
                },
                "WrapperInput": {
                    "type": "object",
                    "title": "WrapperInput",
                    "properties": {
                        "item": {"$ref": REF + "EntityInput"},
                        "items": {"type": "array", "items": {"$ref": REF + "EntityInput"}},
                    },
                },
                "WidgetsPage": {
                    "type": "object",
                    "title": "WidgetsPage",
                    "properties": {"items": {"type": "array"}},
                },
            }
        },
        "paths": {
            "/create": {
                "post": {"requestBody": {"content": {"application/json": {"schema": {"$ref": REF + "FooInput"}}}}}
            },
            "/read": {
                "get": {
                    "responses": {"200": {"content": {"application/json": {"schema": {"$ref": REF + "BarOutput"}}}}}
                }
            },
            "/list": {
                "get": {
                    "responses": {"200": {"content": {"application/json": {"schema": {"$ref": REF + "WidgetsPage"}}}}}
                }
            },
            "/lookup": {"post": {"requestBody": {"content": {"application/json": {"schema": {"$ref": REF + "Baz"}}}}}},
            "/eval": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "discriminator": {
                                        "propertyName": "type",
                                        "mapping": {"m": REF + "Metric"},
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/ref-field": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "anyOf": [
                                        {
                                            "type": "string",
                                            "title": "Reference",
                                            "description": "A reference to Foo.",
                                        },
                                        {"$ref": REF + "FooInput"},
                                    ]
                                }
                            }
                        }
                    }
                }
            },
            "/evaluate": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "discriminator": {
                                        "propertyName": "type",
                                        "mapping": {
                                            "alpha": REF + "AlphaMetric",
                                            "beta": REF + "BetaMetric",
                                        },
                                    }
                                }
                            }
                        }
                    }
                }
            },
        },
    }

    result = tweak_spec(input_spec)
    assert result == expected


def _collision_spec():
    """Two distinct models sharing a class name across modules that normalize to
    the same bare name with *differing* content."""
    return {
        "components": {
            "schemas": {
                "automodel__schema__TrainingSpec": {
                    "type": "object",
                    "properties": {"finetuning_type": {"type": "string"}},
                },
                "unsloth__schemas__TrainingSpec": {
                    "type": "object",
                    "properties": {"use_gradient_checkpointing": {"type": "string"}},
                },
            }
        },
        "paths": {
            "/a": {
                "post": {
                    "requestBody": {
                        "content": {"application/json": {"schema": {"$ref": REF + "automodel__schema__TrainingSpec"}}}
                    }
                }
            }
        },
    }


def test_tweak_spec_raises_on_collision():
    """A differing-content schema-name collision fails the build, rather than
    silently keeping one model and mis-pointing the other's ``$ref``s (which
    would ship a wrong contract in the generated SDK)."""
    with pytest.raises(ValueError, match="schema name collision"):
        tweak_spec(_collision_spec())


def test_tweak_spec_dedups_identical_content_module_qualified_collision():
    """Two module-qualified keys with *identical* content dedup to one schema —
    no raise (the fix must not over-trigger on genuinely-shared shapes)."""
    spec = {
        "components": {
            "schemas": {
                "pkg_a__schema__Shared": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
                "pkg_b__schema__Shared": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
            }
        },
        "paths": {
            "/a": {
                "post": {
                    "requestBody": {
                        "content": {"application/json": {"schema": {"$ref": REF + "pkg_a__schema__Shared"}}}
                    }
                }
            }
        },
    }

    result = tweak_spec(spec)
    assert set(result["components"]["schemas"]) == {"Shared"}
    ref = result["paths"]["/a"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    assert ref == REF + "Shared"


def test_anyof_null_collapse_preserves_format_and_write_only():
    """Collapsing ``anyOf: [SecretStr, null]`` must keep ``format`` / ``writeOnly`` so
    SDK + docs treat the field as sensitive."""
    spec = {
        "components": {
            "schemas": {
                "Secret": {
                    "type": "object",
                    "title": "Secret",
                    "properties": {
                        "value": {
                            "anyOf": [
                                {"type": "string", "format": "password", "writeOnly": True},
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Value",
                            "description": "The new secret value",
                        }
                    },
                }
            }
        },
        "paths": {},
    }

    result = tweak_spec(spec)
    prop = result["components"]["schemas"]["Secret"]["properties"]["value"]
    assert prop == {
        "type": "string",
        "format": "password",
        "writeOnly": True,
        "title": "Value",
        "description": "The new secret value",
    }


def test_anyof_null_collapse_hoists_optional_union():
    """``Optional[Union[...]]`` renders as ``anyOf: [{oneOf: [...]}, null]``.
    Collapsing must drop the null branch and hoist the ``oneOf`` onto the parent."""
    spec = {
        "components": {
            "schemas": {
                "Config": {
                    "type": "object",
                    "title": "Config",
                    "properties": {
                        "replace": {
                            "anyOf": [
                                {
                                    "oneOf": [
                                        {"$ref": "#/components/schemas/Annotate"},
                                        {"$ref": "#/components/schemas/Redact"},
                                    ]
                                },
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Replace",
                        }
                    },
                }
            }
        },
        "paths": {},
    }

    result = tweak_spec(spec)
    prop = result["components"]["schemas"]["Config"]["properties"]["replace"]
    assert prop == {
        "oneOf": [
            {"$ref": "#/components/schemas/Annotate"},
            {"$ref": "#/components/schemas/Redact"},
        ],
        "title": "Replace",
    }


def test_tweak_spec_promotes_schema_default_extension():
    spec = {
        "components": {
            "schemas": {
                "LogQueryRequest": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "x-schema-default": 100,
                        }
                    },
                }
            }
        },
        "paths": {
            "/logs": {
                "get": {
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "schema": {
                                "type": "integer",
                                "x-schema-default": 100,
                            },
                        }
                    ]
                }
            }
        },
    }

    result = tweak_spec(spec)

    query_schema = result["paths"]["/logs"]["get"]["parameters"][0]["schema"]
    model_schema = result["components"]["schemas"]["LogQueryRequest"]["properties"]["limit"]
    assert query_schema["default"] == 100
    assert "x-schema-default" not in query_schema
    assert model_schema["default"] == 100
    assert "x-schema-default" not in model_schema


def test_tweak_spec_removes_null_schema_default_extension_after_nullable_collapse():
    spec = {
        "components": {
            "schemas": {
                "SearchParams": {
                    "type": "object",
                    "properties": {
                        "cursor": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "x-schema-default": None,
                            "title": "Cursor",
                        }
                    },
                }
            }
        },
        "paths": {},
    }

    result = tweak_spec(spec)

    cursor_schema = result["components"]["schemas"]["SearchParams"]["properties"]["cursor"]
    assert cursor_schema == {"type": "string", "title": "Cursor"}
