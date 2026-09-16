<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Guardrail

Types:

```python
from nemo_helix.types.guardrail import (
    ActionRails,
    ActivatedRail,
    AIDefenseRailConfig,
    AutoAlignOptions,
    AutoAlignRailConfig,
    CacheStatsConfig,
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
    ClavataRailConfig,
    ClavataRailOptions,
    ContentSafetyConfig,
    ContextBloatDetectionConfig,
    CrowdStrikeAidrRailConfig,
    DialogRails,
    ExecutedAction,
    FactCheckingRailConfig,
    FiddlerGuardrails,
    Function,
    FunctionCall,
    GLiNERDetection,
    GLiNERDetectionOptions,
    GenerationLog,
    GenerationLogOptions,
    GenerationOptions,
    GenerationRailsOptions,
    GenerationStats,
    GuardrailCheckRequest,
    GuardrailCheckResponse,
    GuardrailConfig,
    GuardrailsAIRailConfig,
    GuardrailsAIValidatorConfig,
    GuardrailsData,
    GuardrailsDataParam,
    ImageURl,
    InjectionDetection,
    InputRails,
    Instruction,
    JailbreakDetectionConfig,
    LLMCallInfo,
    LocalHfClassifierConfig,
    LogAdapterConfig,
    MessageTemplate,
    Model,
    ModelCacheConfig,
    ModelParameters,
    MultilingualConfig,
    OutputRails,
    OutputRailsStreamingConfig,
    PangeaRailConfig,
    PangeaRailOptions,
    PatronusEvaluateAPIParams,
    PatronusEvaluateConfig,
    PatronusEvaluationSuccessStrategy,
    PatronusRailConfig,
    PolygrafDetection,
    PolygrafDetectionOptions,
    PrivateAIDetection,
    PrivateAIDetectionOptions,
    RailStatus,
    Rails,
    RailsConfig,
    RailsConfigData,
    ReasoningConfig,
    RegexDetection,
    RegexDetectionOptions,
    RemoteHfClassifierConfig,
    RetrievalRails,
    SensitiveDataDetection,
    SensitiveDataDetectionOptions,
    SingleCallConfig,
    StatusEnum,
    TaskPrompt,
    ToolInputRails,
    ToolOutputRails,
    TracingConfig,
    TrendMicroRailConfig,
    UserMessagesConfig,
)
```

Methods:

- <code title="post /apis/guardrails/v2/workspaces/{workspace}/checks">client.guardrail.<a href="./src/nemo_helix/resources/guardrail/guardrail.py">check</a>(\*, workspace, \*\*<a href="src/nemo_helix/types/guardrail/guardrail_check_params.py">params</a>) -> <a href="./src/nemo_helix/types/guardrail/guardrail_check_response.py">GuardrailCheckResponse</a></code>

## Configs

Types:

```python
from nemo_helix.types.guardrail import (
    GuardrailConfigFilter,
    GuardrailConfigParam,
    GuardrailConfigUpdate,
    GuardrailConfigsPage,
)
```

Methods:

- <code title="post /apis/guardrails/v2/workspaces/{workspace}/configs">client.guardrail.configs.<a href="./src/nemo_helix/resources/guardrail/configs.py">create</a>(\*, workspace, \*\*<a href="src/nemo_helix/types/guardrail/config_create_params.py">params</a>) -> <a href="./src/nemo_helix/types/guardrail/guardrail_config.py">GuardrailConfig</a></code>
- <code title="get /apis/guardrails/v2/workspaces/{workspace}/configs/{name}">client.guardrail.configs.<a href="./src/nemo_helix/resources/guardrail/configs.py">retrieve</a>(name, \*, workspace) -> <a href="./src/nemo_helix/types/guardrail/guardrail_config.py">GuardrailConfig</a></code>
- <code title="patch /apis/guardrails/v2/workspaces/{workspace}/configs/{name}">client.guardrail.configs.<a href="./src/nemo_helix/resources/guardrail/configs.py">update</a>(name, \*, workspace, \*\*<a href="src/nemo_helix/types/guardrail/config_update_params.py">params</a>) -> <a href="./src/nemo_helix/types/guardrail/guardrail_config.py">GuardrailConfig</a></code>
- <code title="get /apis/guardrails/v2/workspaces/{workspace}/configs">client.guardrail.configs.<a href="./src/nemo_helix/resources/guardrail/configs.py">list</a>(\*, workspace, \*\*<a href="src/nemo_helix/types/guardrail/config_list_params.py">params</a>) -> <a href="./src/nemo_helix/types/guardrail/guardrail_config.py">SyncDefaultPagination[GuardrailConfig]</a></code>
- <code title="delete /apis/guardrails/v2/workspaces/{workspace}/configs/{name}">client.guardrail.configs.<a href="./src/nemo_helix/resources/guardrail/configs.py">delete</a>(name, \*, workspace) -> <a href="./src/nemo_helix/types/shared/delete_response.py">DeleteResponse</a></code>
