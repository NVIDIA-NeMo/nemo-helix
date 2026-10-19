---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: plugin-config
description: Creates plugin configuration using NemoConfig with environment variables and YAML file support. Use when adding plugin configuration fields, reading config values at runtime, setting up test config overrides, or understanding the env var naming formula. Trigger keywords: config, configuration, NemoConfig, env var, environment variable, plugin_name, NHX_CONFIG, YAML config, config override, test config.
---

# Plugin Configuration (NemoConfig)

## Defining Config

Both `plugin_name` AND `plugin_description` are **required** `ClassVar[str]` fields. Omitting either raises `TypeError` at class-definition time.

```python
from typing import ClassVar
from pydantic import Field
from nemo_helix_plugin.config import NemoConfig

class MyPluginConfig(NemoConfig):
    plugin_name: ClassVar[str] = "my-plugin"           # kebab-case — matches CLI entry-point key
    plugin_description: ClassVar[str] = "Configuration for my plugin."

    debug: bool = Field(default=False)
    max_workers: int = Field(default=4)
    log_level: str = Field(default="INFO")
```

## Environment Variable Formula

`NHX_<SAFE_NAME>_<FIELD>` where `SAFE_NAME` = `plugin_name` uppercased with non-word chars replaced by `_`.

| `plugin_name` | Env prefix | Field | Env var |
|---|---|---|---|
| `"my-plugin"` | `NHX_MY_PLUGIN_` | `debug` | `NHX_MY_PLUGIN_DEBUG` |
| `"agents"` | `NHX_AGENTS_` | `runner_backend` | `NHX_AGENTS_RUNNER_BACKEND` |
| `"example"` | `NHX_EXAMPLE_` | `greeting_style` | `NHX_EXAMPLE_GREETING_STYLE` |

Nested fields use `_` as delimiter:
```bash
NHX_AGENTS_CONTROLLER_INTERVAL_SECONDS=10
NHX_AGENTS_CONTROLLER_PORT_RANGE_START=49152
```

## YAML Config File

Config key in `/etc/nhx/config.yaml` equals `plugin_name`:

```yaml
my_plugin:
  debug: true
  max_workers: 8

# Helm value path: platformConfig.my_plugin.debug
```

Override config file location: `NHX_CONFIG_FILE_PATH=/my/custom/config.yaml`

## Priority Order

1. Environment variables (`NHX_MY_PLUGIN_*`) — highest
2. Config file (`/etc/nhx/config.yaml`)
3. Pydantic field defaults — lowest

## Accessing Config

```python
# Option 1: classmethod (preferred in plugin code)
config = MyPluginConfig.get()

# Option 2: standalone function (preferred when importing one name for all config types)
from nemo_helix_plugin.config import get_nemo_config
config = get_nemo_config(MyPluginConfig)
```

Both return the same cached singleton. Multiple calls in the same process return the same instance.

## Example — ExampleConfig from example-plugin (verbatim)

```python
from typing import ClassVar, Literal
from nemo_helix_plugin.config import NemoConfig
from pydantic import Field

class ExampleConfig(NemoConfig):
    plugin_name: ClassVar[str] = "example"
    plugin_description: ClassVar[str] = "Configuration for the NeMo Helix example plugin."

    greeting_style: Literal["formal", "casual"] = Field(
        default="formal",
        description=(
            "Controls the tone of /hello/{name} responses. "
            '"formal" → "Hello, {name}."  "casual" → "Hey, {name}!"'
            "  Set NHX_EXAMPLE_GREETING_STYLE to override."
        ),
    )
    log_requests: bool = Field(
        default=False,
        description=(
            "When True, each request to the items list endpoint emits a "
            "structured INFO log line including the workspace and page number. "
            "Set NHX_EXAMPLE_LOG_REQUESTS=true to enable."
        ),
    )
```

Usage across multiple components (no wiring needed):
```python
# In service.py AND controller.py — both call get() independently
config = ExampleConfig.get()
```

## Nested Config Models

Use a plain `BaseModel` (NOT `NemoConfig`) for nested sections:

```python
from pydantic import BaseModel, Field
from nemo_helix_plugin.config import NemoConfig

class ControllerConfig(BaseModel):           # plain BaseModel — NOT NemoConfig
    interval_seconds: int = Field(default=5)
    health_check_timeout_seconds: int = Field(default=120)

class AgentsConfig(NemoConfig):
    plugin_name: ClassVar[str] = "agents"
    plugin_description: ClassVar[str] = "Configuration for the NeMo Helix agents plugin."

    controller: ControllerConfig = Field(default_factory=ControllerConfig)
    runner_backend: str = Field(default="in_memory")
```

Access nested fields: `AgentsConfig.get().controller.interval_seconds`

Set nested via env: `NHX_AGENTS_CONTROLLER_INTERVAL_SECONDS=10`

## Test Override Pattern

```python
from nemo_helix_plugin.config import (
    set_nemo_config_override,
    clear_nemo_config_override,
    clear_nemo_config_overrides,
)

# In a test:
def test_casual_greeting():
    set_nemo_config_override(ExampleConfig(greeting_style="casual"))
    try:
        config = ExampleConfig.get()
        assert config.greeting_style == "casual"
    finally:
        clear_nemo_config_override(ExampleConfig)

# With autouse pytest fixture (recommended):
import pytest

@pytest.fixture(autouse=True)
def reset_config():
    yield
    clear_nemo_config_overrides()   # clears ALL overrides after each test
```

## Gotchas

- **`plugin_name` must be kebab-case**: `"my-plugin"` not `"my_plugin"`. Use the same value as your CLI entry-point key. The env prefix replaces `-` with `_` automatically.
- **Both `plugin_name` AND `plugin_description` required**: `TypeError` at class-definition time if either is missing or empty.
- **Do NOT call `MyPluginConfig()` directly**: This bypasses env/file loading. Always use `MyPluginConfig.get()` or `get_nemo_config(MyPluginConfig)`.
- **Do NOT set `model_config` manually**: The `_NemoConfigMeta` metaclass sets it before Pydantic builds the schema. Setting it manually will be overridden and may leave the schema inconsistent.
- **Config is cached per process**: In tests, always clean up with `clear_nemo_config_override(cls)` or `clear_nemo_config_overrides()`. Failing to do so pollutes subsequent tests.
- **Abstract intermediates**: A class that declares **neither** `plugin_name` nor `plugin_description` is treated as an abstract intermediate (exempt from validation). Declaring `plugin_description` without `plugin_name` raises `TypeError` — you cannot partially fill the requirement.
