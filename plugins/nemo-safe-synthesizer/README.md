<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Run Safe Synthesizer Jobs

Use the Safe Synthesizer plugin to create jobs through the platform Jobs service.

## Prerequisites

- A running NeMo Helix with Safe Synthesizer, Jobs, Files, and Secrets services available.
- A GPU-capable Jobs backend that can pull the configured Safe Synthesizer task image.
- A Safe Synthesizer job spec, such as `nss-job.json`.
- Platform filesets and secrets referenced by the job spec.

## Steps

1. Start NeMo Helix and confirm readiness:

   ```bash
   curl -s http://localhost:8080/health/ready
   ```

2. Optionally register model filesets:

   ```bash
   uv run python plugins/nemo-safe-synthesizer/scripts/setup_model_filesets.py --files-api-url http://localhost:8080
   ```

3. Create a job through the CLI:

   ```bash
   uv run nemo safe-synthesizer generate \
     --workspace default \
     --spec-file nss-job.json
   ```

   You can also use the SDK builder or Jobs API. See `docs/safe-synthesizer/sdk-resources.mdx`.

## Troubleshooting

- If model downloads fail, confirm the Files API URL is reachable and the model filesets exist in the selected workspace.
- If CUDA initialization fails, inspect the job logs and verify the task image matches the cluster driver/runtime.
- If the job cannot load input data, confirm the fileset reference in the job spec.

## Related Links

- `docs/safe-synthesizer/about/jobs.mdx`
- `docs/safe-synthesizer/about/reference.md`
- `plugins/nemo-safe-synthesizer/scripts/setup_model_filesets.py`

## Next Steps

- Review the architecture reference: `docs/safe-synthesizer/about/reference.md`.
- Run the model setup script: `plugins/nemo-safe-synthesizer/scripts/setup_model_filesets.py`.
- Retrieve job artifacts: `plugins/nemo-safe-synthesizer/src/nemo_safe_synthesizer_plugin/skills/safe-synthesizer/workflows/artifacts.md`.
