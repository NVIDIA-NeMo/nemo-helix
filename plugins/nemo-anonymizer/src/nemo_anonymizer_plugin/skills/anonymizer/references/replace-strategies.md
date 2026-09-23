<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Replace Strategies

Use this reference only for plugin-specific request formatting. The [Anonymizer library docs](https://github.com/NVIDIA-NeMo/Anonymizer/tree/main/docs) and library skills own strategy behavior and parameter details for `Redact`, `Annotate`, `Hash`, `Substitute`, and rewrite mode.

Plugin notes:

- When specifying `config`, choose either `config.replace` or `config.rewrite`, not both.
- Hand-written YAML specs must include a `kind` discriminator inside `replace`.
- Preview/run execution requires `model_configs` so provider calls route through NeMo Helix Inference Gateway.

Minimal YAML shape:

```yaml
config:
  replace:
    kind: redact  # one of: redact, annotate, hash, substitute
    format_template: "[REDACTED_{label}]"
```

For `substitute`, ensure the model pool can satisfy the Anonymizer library `replacement_generator` role. Prefer omitting `selected_models` unless the user specifically asks to pin aliases.
