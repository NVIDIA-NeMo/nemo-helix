<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bedrock plugin

Supplies model metadata for Amazon Bedrock models. It registers the `bedrock` source under the `nemo.models.metadata_sources` entry point, which the Models service's `model-metadata` controller uses for every ModelProvider on a `bedrock-runtime`, `bedrock-runtime-fips`, or `bedrock-mantle` host.

Once per reconcile cycle and region it calls Bedrock's `ListFoundationModels` (`GET https://bedrock.<region>.amazonaws.com/foundation-models`) with the provider's own Bedrock API key, read from the provider's `api_key_secret_name` in Secrets. Cross-region inference profile ids such as `us.anthropic.claude-sonnet-4-6` are matched to their base model. Providers without an API key are skipped.

Fields it fills: display name, input and output modalities, and the embedding capability. Bedrock's catalog has no description, context length, or tool-calling flag.

The API key needs `bedrock:ListFoundationModels` and `bedrock:CallWithBearerToken`. To stop calling Bedrock's catalog, uninstall the plugin or leave `bedrock` out of `NEMO_PLUGIN_MODEL_METADATA_SOURCES_ALLOWLIST`.
