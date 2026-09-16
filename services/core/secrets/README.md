<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Secrets Infrastructure Microservice

The microservice responsible for providing secrets to all the other microservices.

# Local development

If you are using VSCode, you can use this launch configuration to start the Secrets microservice locally with all dependencies:

```json
{
    "configurations": [
        {
            "name": "Debug Platform",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/platform/src/nhx/platform/main.py",
            "args": [
                "run",
                "--services",
                "entities",
                "secrets",
                "auth",
                "--host=0.0.0.0",
                "--port=8000",
            ],
            "env": {
                "NHX_CONFIG_FILE_PATH": "${workspaceFolder}/platform/config/local.yaml",
            },
        }
    ]
}
```

# Creating secrets

A secret name must start with a lowercase letter, end with a lowercase letter or digit, and contain only lowercase letters, digits, and hyphens.

With the SDK, you can test secrets functionality by running:

```python
from nemo_helix import NeMoHelix
sdk = NeMoHelix(base_url="http://localhost:8080")

# Create a secret
secret = sdk.secrets.create(
    name="hf-token",
    workspace="default",
    data="hf_..."
)

# Get a secret's metadata
retrieved_secret = sdk.secrets.retrieve(
    name="hf-token",
    workspace="default",
)


# Access a secret's value
secret_value = sdk.secrets.access(
    name="hf-token",
    workspace="default",
)
hf_token = secret_value.data
```

# Updating Secrets

To update a secret's value, you can update it's data attribute:

```python
# Update the value of the secret
updated_secret = sdk.secrets.update(
    name="hf-token",
    workspace="default",
    data="hf_new_token_..."
)
```

# Deleting Secrets

Deleting a secret will remove it from the platform:

```python
# Delete a secret
sdk.secrets.delete(
    name="hf-token",
    workspace="default"
)
```

# Using Secrets in Jobs

To use secrets in the Jobs API factory, you can define them in the Job compiler when submitting to the Jobs API.

For example:

```python
from nemo_helix import AsyncNeMoHelix
from pydantic import BaseModel
# Import the job compiling building blocks
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    PlatformJobSpec,
    PlatformJobStep,
    EnvironmentVariable,
    EnvironmentVariableFromSecret,
)

sdk = AsyncNeMoHelix(base_url="http://localhost:8080")

class JobConfig(BaseModel):
    # Define your job configuration here
    pass

async def platform_job_compiler(model: JobConfig) -> PlatformJobSpec:
    return PlatformJobSpec(
        steps=[
            PlatformJobStep(
                name="job-using-secrets",
                executor=CPUExecutionProviderSpec(
                    provider="cpu",
                    container=ContainerSpec(...),
                ),
                config=model.model_dump(),
                environment=[
                    # Platform secrets can be referenced directly by name.
                    # We assume a secret named "hf-token" already exists in the same workspace as the job
                    # that we are submitting.
                    EnvironmentVariable(
                        name="HF_TOKEN",
                        from_secret=EnvironmentVariableFromSecret(name="hf-token"),
                    ),
                ],
            ),
        ]
    )
```
