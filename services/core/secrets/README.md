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

With the typed Secrets client, you can test secrets functionality by running:

```python
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest

secrets_client = SecretsClient(base_url="http://localhost:8080", workspace="default")

# Create a secret
secret = secrets_client.create_secret(
    body=HelixSecretCreateRequest(name="hf-token", value="hf_..."),
).data()

# Get a secret's metadata
retrieved_secret = secrets_client.get_secret(name="hf-token").data()

# Access a secret's value
secret_value = secrets_client.access_secret(name="hf-token").data()
hf_token = secret_value.value
```

# Updating Secrets

To update a secret's value or description, send an update request:

```python
from nemo_helix_plugin.secrets.types import HelixSecretUpdateRequest

# Update the value of the secret
updated_secret = secrets_client.update_secret(
    name="hf-token",
    body=HelixSecretUpdateRequest(value="hf_new_token_..."),
).data()
```

# Deleting Secrets

Deleting a secret will remove it from the platform:

```python
# Delete a secret
secrets_client.delete_secret(name="hf-token")
```

# Using Secrets in Jobs

To use secrets in the Jobs API factory, you can define them in the Job compiler when submitting to the Jobs API.

For example:

```python
from pydantic import BaseModel
# Import the job compiling building blocks
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    HelixJobSpec,
    HelixJobStep,
    EnvironmentVariable,
    EnvironmentVariableFromSecret,
)

class JobConfig(BaseModel):
    # Define your job configuration here
    pass

async def platform_job_compiler(model: JobConfig) -> HelixJobSpec:
    return HelixJobSpec(
        steps=[
            HelixJobStep(
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
