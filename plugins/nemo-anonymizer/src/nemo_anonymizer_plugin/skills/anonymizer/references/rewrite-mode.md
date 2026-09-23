<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Rewrite Mode

Use this reference only for plugin execution concerns. The [Anonymizer library docs](https://github.com/NVIDIA-NeMo/Anonymizer/tree/main/docs) and library skills own rewrite semantics, privacy-goal fields, risk tuning, and recommended prompt wording.

For preview/run execution (`sdk.anonymizer.preview`, `nemo anonymizer run`):

- Include `model_configs` so rewrite model calls route through NeMo Helix Inference Gateway providers.
- Use HTTP(S) URLs or fileset references for `data.source`.
- Only include `selected_models.rewrite` when you need to override library defaults, and use Anonymizer library role names exactly.

Example role override shape:

```yaml
model_configs:
  - alias: gliner-pii-detector
    provider: nvidia-build
    model: nvidia/gliner-pii
  - alias: gpt-oss-120b
    provider: nvidia-build
    model: openai/gpt-oss-120b
  - alias: nemotron-30b-thinking
    provider: nvidia-build
    model: nvidia/nemotron-3-nano-30b-a3b

selected_models:
  detection:
    entity_detector: gliner-pii-detector
    entity_validator: gpt-oss-120b
  rewrite:
    rewriter: gpt-oss-120b
    evaluator: nemotron-30b-thinking
    repairer: gpt-oss-120b
    judge: nemotron-30b-thinking
```

Prefer omitting `selected_models` unless the user specifically asks to pin aliases.
