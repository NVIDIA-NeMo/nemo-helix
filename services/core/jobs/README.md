<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Jobs Infrastructure Microservice

A microservice for providing a generic job runner for NeMo Helix.

- [Integrating a functional microservice with Jobs](#integrating-a-functional-microservice-with-jobs)
- [Local Development](#local-development)

## Integrating a functional microservice with Jobs

To integrate a functional microservice with the Jobs microservice, there are three steps:

1. [Create a Jobs configuration compiler](#create-a-jobs-configuration-compiler)
2. [Implement the Jobs API Factory](#implement-the-jobs-api-factory)
3. [Schedule a Job](#schedule-a-job)

After this is done, you should be able to schedule jobs against the Jobs microservice from your functional microservice.

### Create a Jobs configuration compiler

Each functional microservice will be responsible for translating their specific job's request object into a `PlatformJobSpec` that can be submitted to the Jobs API. As defined in the [Jobs RFC API Interfaces](https://docs.google.com/document/d/1KhF0ED9OGhFIHu8-wittuhMHWOBvNvBQuqz3ahk63vY/edit?tab=t.0#heading=h.ccl0b5irew8u), every request into a functional microservice will satisfy a base JobRequest schema that includes a `spec` field that is customizable. This field is then used to compile a platform Job specification. The compiler is provided by the functional microservice, leveraging SDK-provided objects to build a platform job.

Suppose you have your functional microservice's job specification as below:

```python
from pydantic import BaseModel
from typing import Optional

class MyFunctionalJobConfig(BaseModel):
  dataset: str
  target: str
  config: dict
  memory: Optional[str] = None
  cpus: Optional[str] = None
   # The execution profile allows a job to select the hardware based on available execution profiles
  execution_profile: Optional[str] = None
```

You can create a platform job configuration compiler like the following:

- NOTE: Environment variables and secrets guidance is TBD

```python
## Import the necessary building blocks from the factory
from nemo_helix_plugin.jobs.api_factory import (
    PlatformJobSpec,
    PlatformJobStep,
    CPUExecutionProviderSpec,
    ContainerSpec,
    ResourcesSpec,
    ResourcesRequestsSpec,
    ResourcesLimitsSpec
)

## You may supply some values for jobs based on configuration values from your functional microservice.
# This will be determined by the functional microservice on a service-by-service basis.
from pydantic_settings import BaseSettings

class MyFunctionalMicroserviceSettings(BaseSettings):
    job_image: str = Field(default="nvcr.io/nvidia/nemo-helix/my-functional-microservice:v0.0.1")
    job_command: list[str] = Field(default=[])
    job_args: list[str] = Field(default=["--target", "default"])
    default_job_resource_cpu_request: str = Field(default="1")
    default_job_resource_memory_request: str = Field(default="1Gi")
    default_job_resource_cpu_limit: str = Field(default="1")
    default_job_resource_memory_limit: str = Field(default="1Gi")

settings = MyFunctionalMicroserviceSettings()

## Create the config compiler
def my_functional_job_compiler(model: MyFunctionalJobConfig) -> PlatformJobSpec:

    # Create the job step's resources from settings
    resources = ResourcesSpec(
        limits=ResourcesLimitsSpec(
            memory=settings.default_job_resource_memory_limit,
            cpu=settings.default_job_resource_cpu_limit,
        ),
        # A functional microservice's job config might supply overrides to default values
        requests=ResourcesRequestsSpec(
            memory=model.memory or settings.default_job_resource_memory_request,
            cpu=model.cpu or settings.default_job_resource_cpu_request,
        ),
    )

    # Create the job steps based on settings and the job request's spec
    return PlatformJobSpec(
        steps=[
            PlatformJobStep(
                name="my-job-step-1",
                executor=CPUExecutionProviderSpec(
                    profile=model.execution_profile,
                    container=ContainerSpec(
                      image=settings.job_image,
                      command=settings.job_command,
                      args=settings.job_args,
                    ),
                    resources=resources,
                ),
                config=model.model_dump(),
                environment={"ENV_VAR": "test_value"},
            ),
            # Multiple steps can be configured
            PlatformJobStep(
                name="my-job-step-2",
                executor=CPUExecutionProviderSpec(
                    profile=model.execution_profile,
                    container=ContainerSpec(image=settings.job_image),
                    resources=resources,
                ),
                config=model.model_dump(),
                environment={"ENV_VAR": "test_value"},
            ),
        ]
    )
```

### Implement the Jobs API Factory

Now that you have a platform job configuration compiler, you can implement the Jobs api factory as follows:

```python
from nemo_helix_plugin.jobs.api_factory import job_route_factory

service_name = "my-functional-microservice"
jobs_router = job_route_factory(
    service_name=service_name,
    job_type="MyFunctionalMicroservice",
    job_config=MyFunctionalJobConfig,
    platform_job_config_compiler=my_functional_job_compiler,
)

app = FastAPI()
app.include_router(jobs_router, prefix=f"/v1/{service_name}", tags=["My Functional Microservice"])
```

This will create all the necessary job routes prefixed with `/v1/my-functional-microservice`.

Note the following:

- The `service_name` should match the path used by the functional microservice.
- The `job_type` will generate an OpenAPI schema called `MyFunctionalMicroserviceJobRequest`
- The `job_config` is the model for the `spec` attribute of the request, not the entire API request object.

### Schedule a job

Now that you have your microservice integration, you can test this with a curl command against your functional microservice.

Assuming you have a payload like the following in `payload.json`:

```json
{
  "name": "your-job-name",
  "description": "Your job description",
  "project": "proj-1234",
  "spec": { # Your functional microservice's specific job specification
    "dataset": "my-dataset-id-1234",
    "target": "my-target-config",
    "config": {...}
  },
  "ownership": {...},
  "custom_fields": {
    "optional_custom_field_1": "blah1234"
  }
}
```

You can create a job with your functional microservice using this command, assuming your microservice is running locally on port 9000:

```bash
curl -X POST -d @payload.json http://localhost:9000/v1/my-functional-microservice/jobs
```

This should return a job response object that includes the created job's id.

```json
{
  "id": "job-2viu3Vkq1boX2fDzqxpUPY",
  "name": "your-job-name",
  "description": "Your job description",
  "project": "proj-1234",
  "created_at": "2025-08-19T18:29:58.213991",
  "updated_at": "2025-08-19T18:29:58.213996",
  "spec": {
    "dataset": "my-dataset-id-1234",
    "target": "my-target-config",
    "config": {...}
  },
  "status": "created",
  "status_details": {},
  "ownership": {...},
  "custom_fields": {
    "optional_custom_field_1": "blah1234"
  }
}
```

You can also verify the job exists on the Jobs microservice directly. Assuming you are running the jobs api on port 8000, you can query for the job as follows, including the generated platform configuration.

```bash
curl http://localhost:8080/v1/jobs/job-some-random-id
```

```json
{
  "id": "job-2viu3Vkq1boX2fDzqxpUPY",
  "name": "your-job-name",
  "description": "Your job description",
  "project": "proj-1234",
  "created_at": "2025-08-19T18:29:58.213991",
  "updated_at": "2025-08-19T18:29:58.213996",
  "source": "my-functional-microservice",
  "spec": {
    "dataset": "my-dataset-id-1234",
    "target": "my-target-config",
    "config": {...}
  },
  "platform_spec": {
    "steps": [
      {
        "name": "my-job-step-1",
        "environment": {
          "ENV_VAR": "test_value"
        },
        "executor": {
          "provider": "cpu",
          "profile": "default",
          "container": {
            "image": "nvcr.io/nvidia/nemo-helix/my-functional-microservice:v0.0.1",
            "command": [],
            "args": ["--target", "default"]
          },
          "resources": {
            "requests": {
              "cpu": "1",
              "memory": "1Gi"
            },
            "limits": {
              "cpu": "1",
              "memory": "1Gi"
            }
          }
        },
        "config": {...}
      },
      {
        "name": "my-job-step-2",
        "environment": {
          "ENV_VAR": "test_value"
        },
        "executor": {
          "provider": "cpu",
          "profile": "default",
          "container": {
            "image": "nvcr.io/nvidia/nemo-helix/my-functional-microservice:v0.0.1"
          },
          "resources": {
            "requests": {
              "cpu": "1",
              "memory": "1Gi"
            },
            "limits": {
              "cpu": "1",
              "memory": "1Gi"
            }
          }
        },
        "config": {...}
      }
    ]
  },
  "status": "created",
  "status_details": {},
  "ownership": {...},
  "custom_fields": {
    "optional_custom_field_1": "blah1234"
  }
}
```

## Local development

### Requirements

Currently jobs only supports Postgres as the backend.

```
# Start Postgres and Fluentbit from the Quickstart
cd services/core/infrastructure/jobs
(cd ../../deploy/quickstart/external && docker compose --env-file ../dev_overrides/env/local-registry.env up fluentbit postgres datastore openbao --wait)
```

### Start the API server

```
DATABASE_HOST=localhost NHX_CONFIG_FILE_PATH=config/local.yaml DEBUG=True uv run python -m jobs.api.server
```

### Start the controller

```
DATABASE_HOST=localhost NHX_CONFIG_FILE_PATH=config/local.yaml DEBUG=True uv run python -m jobs.controller.main
```

### Start both API server and controller

A convenience script is provided to start both servers:

```
./run.sh
```

Use the following curl command to launch a local docker based job

```bash
curl -X POST "http://localhost:8080/v1/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "docker-test-job",
    "source": "curl-test",
    "spec": {
      "parameters": {
        "test_param": "test_value"
      }
    },
    "platform_spec": {
      "steps": [
        {
          "name": "docker-step",
          "executor": {
            "provider": "cpu",
            "profile": "default",
            "container": {
              "image": "hello-world"
            }
          },
          "config": {
            "command": "echo",
            "args": ["Hello from docker job!"]
          },
          "environment": {
            "TEST_ENV": "test_value"
          }
        }
      ]
    }
  }'
```

To create a job and tail a log, there are some scripts to help test:

```
# Paging
uv run python script/test_e2e.py start_and_watch
```

### Subprocess backend

The `subprocess/default` profile is registered by default for non-Kubernetes runtimes, including local Docker development. It runs the job command as a host subprocess and emits stdout/stderr through the normal Files/OTEL-backed Jobs logs API.

```bash
JOB_NAME="subprocess-$(date +%s)"

curl -sS -X POST "http://localhost:8080/apis/jobs/v2/workspaces/default/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "'"$JOB_NAME"'",
    "source": "curl-subprocess-test",
    "spec": {},
    "platform_spec": {
      "steps": [
        {
          "name": "subprocess-step",
          "executor": {
            "provider": "subprocess",
            "profile": "default",
            "command": ["/bin/sh", "-c", "echo hello-subprocess; echo workspace=$NEMO_JOB_WORKSPACE; echo job=$NEMO_JOB_ID"]
          },
          "config": {}
        }
      ]
    }
  }'

curl -sS "http://localhost:8080/apis/jobs/v2/workspaces/default/jobs/$JOB_NAME" \
  | jq '{name, status, status_details}'

curl -sS "http://localhost:8080/apis/jobs/v2/workspaces/default/jobs/$JOB_NAME/logs" \
  | jq '{total, messages: [.data[].message]}'
```

## Local Development with Docker

```
cd ../..deploy/quickstart
docker compose -f internal/docker-compose.yaml --env-file internal/env/dev.env up jobs-controller jobs-api envoy-gateway fluentbit
```

If using colima, you may need to point at the correct docker daemon with the following:

```
export DOCKER_HOST=unix:///Users/rsadler/.colima/default/docker.sock
```

Also ensure that your buildx is using the `docker` driver, and not the `docker-container` driver, as it needs to connect directly to the docker daemon.  To check your settings:

```
docker buildx ls
```

## Local Development with Kubernetes

- Install kind: https://kind.sigs.k8s.io/docs/user/quick-start/#installation
- Create a kind cluster with registry support using `bash /script/kind-with-registry.sh`
- Install Skaffold: https://skaffold.dev/docs/install/
- Update dependencies of the Helm chart using `helm dep update k8s/helm`
- Run skaffold: `skaffold dev --default-repo=localhost:5001 --keep-running-on-failure`
- If port forward doesn't work, try manually portforwarding: `kubectl port-forward service/nemo-core-api 8000:8000`

## E2E Testing

To run end to end tests on the jobs service, first start the service either using quickstart, or skaffold.

Once the service has launched, you can run tests as follows:

```
uv run --frozen pytest script/test_e2e.py -s
```

You can also exercise the jobs API using this same script as follows:

```

# pause a job
uv run --frozen python script/test_e2e.py pause JOB_ID

# start a test job with the service running on port 8080
uv run --frozen python script/test_e2e.py start --url http://localhost:8080

# tail a job running at http://production:8000
uv run --frozen python script/test_e2e.py tail JOB_ID --url http://production:8000
```
