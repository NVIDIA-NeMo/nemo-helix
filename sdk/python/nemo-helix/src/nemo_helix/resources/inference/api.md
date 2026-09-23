<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Inference

## VirtualModels

Types:

```python
from nemo_helix.types.inference import (
    CreateVirtualModelRequest,
    MiddlewareCall,
    UpdateVirtualModelRequest,
    VirtualModel,
    VirtualModelFilter,
    VirtualModelInferenceConfig,
    VirtualModelsPage,
)
```

Methods:

- <code title="post /apis/inference-gateway/v2/workspaces/{workspace}/virtual-models">client.inference.virtual_models.<a href="./src/nemo_helix/resources/inference/virtual_models.py">create</a>(\*, workspace, \*\*<a href="src/nemo_helix/types/inference/virtual_model_create_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/virtual_model.py">VirtualModel</a></code>
- <code title="get /apis/inference-gateway/v2/workspaces/{workspace}/virtual-models/{name}">client.inference.virtual_models.<a href="./src/nemo_helix/resources/inference/virtual_models.py">retrieve</a>(name, \*, workspace) -> <a href="./src/nemo_helix/types/inference/virtual_model.py">VirtualModel</a></code>
- <code title="get /apis/inference-gateway/v2/workspaces/{workspace}/virtual-models">client.inference.virtual_models.<a href="./src/nemo_helix/resources/inference/virtual_models.py">list</a>(\*, workspace, \*\*<a href="src/nemo_helix/types/inference/virtual_model_list_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/virtual_model.py">SyncDefaultPagination[VirtualModel]</a></code>
- <code title="delete /apis/inference-gateway/v2/workspaces/{workspace}/virtual-models/{name}">client.inference.virtual_models.<a href="./src/nemo_helix/resources/inference/virtual_models.py">delete</a>(name, \*, workspace, \*\*<a href="src/nemo_helix/types/inference/virtual_model_delete_params.py">params</a>) -> None</code>
- <code title="patch /apis/inference-gateway/v2/workspaces/{workspace}/virtual-models/{name}">client.inference.virtual_models.<a href="./src/nemo_helix/resources/inference/virtual_models.py">patch</a>(name, \*, workspace, \*\*<a href="src/nemo_helix/types/inference/virtual_model_patch_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/virtual_model.py">VirtualModel</a></code>

## Models

Types:

```python
from nemo_helix.types.inference import OpenAIListModelsResp, OpenAIModelResp
```

Methods:

- <code title="get /apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/models">client.inference.models.<a href="./src/nemo_helix/resources/inference/models.py">list</a>(\*, workspace) -> <a href="./src/nemo_helix/types/inference/gateway/openai/v1/openai_list_models_resp.py">OpenAIListModelsResp</a></code>
- <code title="get /apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/models/{name}">client.inference.models.<a href="./src/nemo_helix/resources/inference/models.py">get</a>(name, \*, workspace) -> <a href="./src/nemo_helix/types/inference/gateway/openai/v1/openai_model_resp.py">OpenAIModelResp</a></code>

## DeploymentConfigs

Types:

```python
from nemo_helix.types.inference import (
    ContainerExecutorConfig,
    CreateModelDeploymentConfigRequest,
    Engine,
    K8sNIMOperatorConfig,
    ModelDeploymentConfig,
    ModelDeploymentConfigFilter,
    ModelDeploymentConfigModelSpec,
    ModelDeploymentConfigsPage,
    ModelType,
    UpdateModelDeploymentConfigRequest,
)
```

Methods:

- <code title="post /apis/models/v2/workspaces/{workspace}/deployment-configs">client.inference.deployment_configs.<a href="./src/nemo_helix/resources/inference/deployment_configs/deployment_configs.py">create</a>(\*, workspace, \*\*<a href="src/nemo_helix/types/inference/deployment_config_create_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/model_deployment_config.py">ModelDeploymentConfig</a></code>
- <code title="get /apis/models/v2/workspaces/{workspace}/deployment-configs/{name}">client.inference.deployment_configs.<a href="./src/nemo_helix/resources/inference/deployment_configs/deployment_configs.py">retrieve</a>(name, \*, workspace) -> <a href="./src/nemo_helix/types/inference/model_deployment_config.py">ModelDeploymentConfig</a></code>
- <code title="post /apis/models/v2/workspaces/{workspace}/deployment-configs/{name}">client.inference.deployment_configs.<a href="./src/nemo_helix/resources/inference/deployment_configs/deployment_configs.py">update</a>(name, \*, workspace, \*\*<a href="src/nemo_helix/types/inference/deployment_config_update_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/model_deployment_config.py">ModelDeploymentConfig</a></code>
- <code title="get /apis/models/v2/workspaces/{workspace}/deployment-configs">client.inference.deployment_configs.<a href="./src/nemo_helix/resources/inference/deployment_configs/deployment_configs.py">list</a>(\*, workspace, \*\*<a href="src/nemo_helix/types/inference/deployment_config_list_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/model_deployment_config.py">SyncDefaultPagination[ModelDeploymentConfig]</a></code>
- <code title="delete /apis/models/v2/workspaces/{workspace}/deployment-configs/{name}">client.inference.deployment_configs.<a href="./src/nemo_helix/resources/inference/deployment_configs/deployment_configs.py">delete</a>(name, \*, workspace) -> None</code>

### Versions

Types:

```python
from nemo_helix.types.inference.deployment_configs import VersionListResponse
```

Methods:

- <code title="get /apis/models/v2/workspaces/{workspace}/deployment-configs/{config}/versions/{name}">client.inference.deployment_configs.versions.<a href="./src/nemo_helix/resources/inference/deployment_configs/versions.py">retrieve</a>(name, \*, workspace, config) -> <a href="./src/nemo_helix/types/inference/model_deployment_config.py">ModelDeploymentConfig</a></code>
- <code title="get /apis/models/v2/workspaces/{workspace}/deployment-configs/{name}/versions">client.inference.deployment_configs.versions.<a href="./src/nemo_helix/resources/inference/deployment_configs/versions.py">list</a>(name, \*, workspace) -> <a href="./src/nemo_helix/types/inference/deployment_configs/version_list_response.py">VersionListResponse</a></code>
- <code title="delete /apis/models/v2/workspaces/{workspace}/deployment-configs/{config}/versions/{name}">client.inference.deployment_configs.versions.<a href="./src/nemo_helix/resources/inference/deployment_configs/versions.py">delete</a>(name, \*, workspace, config) -> None</code>

## Deployments

Types:

```python
from nemo_helix.types.inference import (
    CreateModelDeploymentRequest,
    ModelDeployment,
    ModelDeploymentFilter,
    ModelDeploymentStatus,
    ModelDeploymentStatusHistoryItem,
    ModelDeploymentsPage,
    UpdateModelDeploymentRequest,
    UpdateModelDeploymentStatusRequest,
    DeploymentListModelsResponse,
)
```

Methods:

- <code title="post /apis/models/v2/workspaces/{workspace}/deployments">client.inference.deployments.<a href="./src/nemo_helix/resources/inference/deployments/deployments.py">create</a>(\*, workspace, \*\*<a href="src/nemo_helix/types/inference/deployment_create_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/model_deployment.py">ModelDeployment</a></code>
- <code title="get /apis/models/v2/workspaces/{workspace}/deployments/{name}">client.inference.deployments.<a href="./src/nemo_helix/resources/inference/deployments/deployments.py">retrieve</a>(name, \*, workspace) -> <a href="./src/nemo_helix/types/inference/model_deployment.py">ModelDeployment</a></code>
- <code title="post /apis/models/v2/workspaces/{workspace}/deployments/{name}">client.inference.deployments.<a href="./src/nemo_helix/resources/inference/deployments/deployments.py">update</a>(name, \*, workspace, \*\*<a href="src/nemo_helix/types/inference/deployment_update_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/model_deployment.py">ModelDeployment</a></code>
- <code title="get /apis/models/v2/workspaces/{workspace}/deployments">client.inference.deployments.<a href="./src/nemo_helix/resources/inference/deployments/deployments.py">list</a>(\*, workspace, \*\*<a href="src/nemo_helix/types/inference/deployment_list_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/model_deployment.py">SyncDefaultPagination[ModelDeployment]</a></code>
- <code title="delete /apis/models/v2/workspaces/{workspace}/deployments/{name}">client.inference.deployments.<a href="./src/nemo_helix/resources/inference/deployments/deployments.py">delete</a>(name, \*, workspace) -> object</code>
- <code title="get /apis/models/v2/workspaces/{workspace}/deployments/{name}/models">client.inference.deployments.<a href="./src/nemo_helix/resources/inference/deployments/deployments.py">list_models</a>(name, \*, workspace) -> <a href="./src/nemo_helix/types/inference/deployment_list_models_response.py">DeploymentListModelsResponse</a></code>
- <code title="post /apis/models/v2/workspaces/{workspace}/deployments/{name}/status">client.inference.deployments.<a href="./src/nemo_helix/resources/inference/deployments/deployments.py">update_status</a>(name, \*, workspace, \*\*<a href="src/nemo_helix/types/inference/deployment_update_status_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/model_deployment.py">ModelDeployment</a></code>

### Versions

Types:

```python
from nemo_helix.types.inference.deployments import VersionListResponse
```

Methods:

- <code title="get /apis/models/v2/workspaces/{workspace}/deployments/{deployment}/versions/{name}">client.inference.deployments.versions.<a href="./src/nemo_helix/resources/inference/deployments/versions.py">retrieve</a>(name, \*, workspace, deployment) -> <a href="./src/nemo_helix/types/inference/model_deployment.py">ModelDeployment</a></code>
- <code title="get /apis/models/v2/workspaces/{workspace}/deployments/{name}/versions">client.inference.deployments.versions.<a href="./src/nemo_helix/resources/inference/deployments/versions.py">list</a>(name, \*, workspace) -> <a href="./src/nemo_helix/types/inference/deployments/version_list_response.py">VersionListResponse</a></code>
- <code title="delete /apis/models/v2/workspaces/{workspace}/deployments/{deployment}/versions/{name}">client.inference.deployments.versions.<a href="./src/nemo_helix/resources/inference/deployments/versions.py">delete</a>(name, \*, workspace, deployment) -> object</code>

## Providers

Types:

```python
from nemo_helix.types.inference import (
    CreateModelProviderRequest,
    ModelProvider,
    ModelProviderFilter,
    ModelProviderSort,
    ModelProviderStatus,
    ModelProvidersPage,
    ServedModelMapping,
    UpdateModelProviderStatusRequest,
    UpsertModelProviderRequest,
)
```

Methods:

- <code title="post /apis/models/v2/workspaces/{workspace}/providers">client.inference.providers.<a href="./src/nemo_helix/resources/inference/providers.py">create</a>(\*, workspace, \*\*<a href="src/nemo_helix/types/inference/provider_create_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/model_provider.py">ModelProvider</a></code>
- <code title="get /apis/models/v2/workspaces/{workspace}/providers/{name}">client.inference.providers.<a href="./src/nemo_helix/resources/inference/providers.py">retrieve</a>(name, \*, workspace) -> <a href="./src/nemo_helix/types/inference/model_provider.py">ModelProvider</a></code>
- <code title="put /apis/models/v2/workspaces/{workspace}/providers/{name}">client.inference.providers.<a href="./src/nemo_helix/resources/inference/providers.py">update</a>(name, \*, workspace, \*\*<a href="src/nemo_helix/types/inference/provider_update_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/model_provider.py">ModelProvider</a></code>
- <code title="get /apis/models/v2/workspaces/{workspace}/providers">client.inference.providers.<a href="./src/nemo_helix/resources/inference/providers.py">list</a>(\*, workspace, \*\*<a href="src/nemo_helix/types/inference/provider_list_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/model_provider.py">SyncDefaultPagination[ModelProvider]</a></code>
- <code title="delete /apis/models/v2/workspaces/{workspace}/providers/{name}">client.inference.providers.<a href="./src/nemo_helix/resources/inference/providers.py">delete</a>(name, \*, workspace) -> None</code>
- <code title="put /apis/models/v2/workspaces/{workspace}/providers/{name}/status">client.inference.providers.<a href="./src/nemo_helix/resources/inference/providers.py">update_status</a>(name, \*, workspace, \*\*<a href="src/nemo_helix/types/inference/provider_update_status_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/model_provider.py">ModelProvider</a></code>

## Prompts

Types:

```python
from nemo_helix.types.inference import (
    ChatCompletionTool,
    CreatePromptRequest,
    FunctionDefinition,
    Prompt,
    PromptFilter,
    PromptMessage,
    PromptMessageRole,
    PromptSort,
    PromptsPage,
    UpdatePromptRequest,
)
```

Methods:

- <code title="post /apis/models/v2/workspaces/{workspace}/prompts">client.inference.prompts.<a href="./src/nemo_helix/resources/inference/prompts.py">create</a>(\*, workspace, \*\*<a href="src/nemo_helix/types/inference/prompt_create_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/prompt.py">Prompt</a></code>
- <code title="get /apis/models/v2/workspaces/{workspace}/prompts/{name}">client.inference.prompts.<a href="./src/nemo_helix/resources/inference/prompts.py">retrieve</a>(name, \*, workspace) -> <a href="./src/nemo_helix/types/inference/prompt.py">Prompt</a></code>
- <code title="put /apis/models/v2/workspaces/{workspace}/prompts/{name}">client.inference.prompts.<a href="./src/nemo_helix/resources/inference/prompts.py">update</a>(name, \*, workspace, \*\*<a href="src/nemo_helix/types/inference/prompt_update_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/prompt.py">Prompt</a></code>
- <code title="get /apis/models/v2/workspaces/{workspace}/prompts">client.inference.prompts.<a href="./src/nemo_helix/resources/inference/prompts.py">list</a>(\*, workspace, \*\*<a href="src/nemo_helix/types/inference/prompt_list_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/prompt.py">SyncDefaultPagination[Prompt]</a></code>
- <code title="delete /apis/models/v2/workspaces/{workspace}/prompts/{name}">client.inference.prompts.<a href="./src/nemo_helix/resources/inference/prompts.py">delete</a>(name, \*, workspace) -> None</code>

## Gateway

### OpenAI

Types:

```python
from nemo_helix.types.inference.gateway import (
    OpenAIPatchResponse,
    OpenAIPostResponse,
    OpenAIPutResponse,
)
```

Methods:

- <code title="delete /apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}">client.inference.gateway.openai.<a href="./src/nemo_helix/resources/inference/gateway/openai/openai.py">delete</a>(trailing_uri, \*, workspace) -> object</code>
- <code title="get /apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}">client.inference.gateway.openai.<a href="./src/nemo_helix/resources/inference/gateway/openai/openai.py">get</a>(trailing_uri, \*, workspace) -> object</code>
- <code title="patch /apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}">client.inference.gateway.openai.<a href="./src/nemo_helix/resources/inference/gateway/openai/openai.py">patch</a>(trailing_uri, \*, workspace, \*\*<a href="src/nemo_helix/types/inference/gateway/openai_patch_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/gateway/openai_patch_response.py">OpenAIPatchResponse</a></code>
- <code title="post /apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}">client.inference.gateway.openai.<a href="./src/nemo_helix/resources/inference/gateway/openai/openai.py">post</a>(trailing_uri, \*, workspace, \*\*<a href="src/nemo_helix/types/inference/gateway/openai_post_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/gateway/openai_post_response.py">OpenAIPostResponse</a></code>
- <code title="put /apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}">client.inference.gateway.openai.<a href="./src/nemo_helix/resources/inference/gateway/openai/openai.py">put</a>(trailing_uri, \*, workspace, \*\*<a href="src/nemo_helix/types/inference/gateway/openai_put_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/gateway/openai_put_response.py">OpenAIPutResponse</a></code>

#### V1

##### Models

Types:

```python
from nemo_helix.types.inference.gateway.openai.v1 import OpenAIListModelsResp, OpenAIModelResp
```

Methods:

- <code title="get /apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/models">client.inference.gateway.openai.v1.models.<a href="./src/nemo_helix/resources/inference/gateway/openai/v1/models.py">list</a>(\*, workspace) -> <a href="./src/nemo_helix/types/inference/gateway/openai/v1/openai_list_models_resp.py">OpenAIListModelsResp</a></code>
- <code title="get /apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/models/{name}">client.inference.gateway.openai.v1.models.<a href="./src/nemo_helix/resources/inference/gateway/openai/v1/models.py">get</a>(name, \*, workspace) -> <a href="./src/nemo_helix/types/inference/gateway/openai/v1/openai_model_resp.py">OpenAIModelResp</a></code>

### Model

Types:

```python
from nemo_helix.types.inference.gateway import (
    ModelPatchResponse,
    ModelPostResponse,
    ModelPutResponse,
)
```

Methods:

- <code title="delete /apis/inference-gateway/v2/workspaces/{workspace}/model/{name}/-/{trailing_uri}">client.inference.gateway.model.<a href="./src/nemo_helix/resources/inference/gateway/model.py">delete</a>(trailing_uri, \*, workspace, name) -> object</code>
- <code title="get /apis/inference-gateway/v2/workspaces/{workspace}/model/{name}/-/{trailing_uri}">client.inference.gateway.model.<a href="./src/nemo_helix/resources/inference/gateway/model.py">get</a>(trailing_uri, \*, workspace, name) -> object</code>
- <code title="patch /apis/inference-gateway/v2/workspaces/{workspace}/model/{name}/-/{trailing_uri}">client.inference.gateway.model.<a href="./src/nemo_helix/resources/inference/gateway/model.py">patch</a>(trailing_uri, \*, workspace, name, \*\*<a href="src/nemo_helix/types/inference/gateway/model_patch_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/gateway/model_patch_response.py">ModelPatchResponse</a></code>
- <code title="post /apis/inference-gateway/v2/workspaces/{workspace}/model/{name}/-/{trailing_uri}">client.inference.gateway.model.<a href="./src/nemo_helix/resources/inference/gateway/model.py">post</a>(trailing_uri, \*, workspace, name, \*\*<a href="src/nemo_helix/types/inference/gateway/model_post_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/gateway/model_post_response.py">ModelPostResponse</a></code>
- <code title="put /apis/inference-gateway/v2/workspaces/{workspace}/model/{name}/-/{trailing_uri}">client.inference.gateway.model.<a href="./src/nemo_helix/resources/inference/gateway/model.py">put</a>(trailing_uri, \*, workspace, name, \*\*<a href="src/nemo_helix/types/inference/gateway/model_put_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/gateway/model_put_response.py">ModelPutResponse</a></code>

### Provider

Types:

```python
from nemo_helix.types.inference.gateway import (
    ProviderPatchResponse,
    ProviderPostResponse,
    ProviderPutResponse,
    ProviderReadyResponse,
)
```

Methods:

- <code title="delete /apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri}">client.inference.gateway.provider.<a href="./src/nemo_helix/resources/inference/gateway/provider.py">delete</a>(trailing_uri, \*, workspace, name) -> object</code>
- <code title="get /apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri}">client.inference.gateway.provider.<a href="./src/nemo_helix/resources/inference/gateway/provider.py">get</a>(trailing_uri, \*, workspace, name) -> object</code>
- <code title="patch /apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri}">client.inference.gateway.provider.<a href="./src/nemo_helix/resources/inference/gateway/provider.py">patch</a>(trailing_uri, \*, workspace, name, \*\*<a href="src/nemo_helix/types/inference/gateway/provider_patch_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/gateway/provider_patch_response.py">ProviderPatchResponse</a></code>
- <code title="post /apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri}">client.inference.gateway.provider.<a href="./src/nemo_helix/resources/inference/gateway/provider.py">post</a>(trailing_uri, \*, workspace, name, \*\*<a href="src/nemo_helix/types/inference/gateway/provider_post_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/gateway/provider_post_response.py">ProviderPostResponse</a></code>
- <code title="put /apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri}">client.inference.gateway.provider.<a href="./src/nemo_helix/resources/inference/gateway/provider.py">put</a>(trailing_uri, \*, workspace, name, \*\*<a href="src/nemo_helix/types/inference/gateway/provider_put_params.py">params</a>) -> <a href="./src/nemo_helix/types/inference/gateway/provider_put_response.py">ProviderPutResponse</a></code>
- <code title="get /apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/ready">client.inference.gateway.provider.<a href="./src/nemo_helix/resources/inference/gateway/provider.py">ready</a>(name, \*, workspace) -> <a href="./src/nemo_helix/types/inference/gateway/provider_ready_response.py">ProviderReadyResponse</a></code>
