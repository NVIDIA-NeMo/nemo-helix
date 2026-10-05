<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Running Safe Synthesizer

## Platform Job Prerequisites

- The Safe Synthesizer service, Jobs service, and GPU execution backend are available.
- `data_source` points to a platform fileset path such as `default/my-fileset#input.csv`.
- If `hf_token_secret` is set, the named platform secret exists in the target workspace.
- If PII classification is enabled, the model provider exists and is referenced as `<workspace>/<provider_name>`.
- The Jobs backend can pull the configured Safe Synthesizer task image.

For platform jobs, start services first (`nemo setup --start-services` or `nemo services run`), then optionally register model filesets:

```bash
curl -s http://localhost:8080/health/ready
uv run python plugins/nemo-safe-synthesizer/scripts/setup_model_filesets.py --files-api-url http://localhost:8080
```

## Container Image Configuration

Prefer the released task image from NGC:

```bash
export NHX_IMAGE_REGISTRY=nvcr.io/nvidia/nemo-helix
export NHX_IMAGE_TAG=<tag>  # match your installed NeMo Helix release
export NEMO_SAFE_SYNTHESIZER_CONTAINER_IMAGE=nhx-safe-synthesizer-tasks
```

This resolves platform job steps to `nvcr.io/nvidia/nemo-helix/nhx-safe-synthesizer-tasks:<tag>`.

For a local Docker-built image on a Docker executor, set a full image reference override:

```bash
docker buildx bake nhx-safe-synthesizer-tasks-docker
export NEMO_SAFE_SYNTHESIZER_CONTAINER_IMAGE_REF=nhx-safe-synthesizer-tasks:local
```

For Kubernetes, push the local build to a registry the cluster can pull, then set `NEMO_SAFE_SYNTHESIZER_CONTAINER_IMAGE_REF` to that pushed image reference.

## Choose the Execution Mode

Use platform jobs for Safe Synthesizer usage. The platform compiles the spec into a GPU container step that runs the configured Safe Synthesizer task image.

Use the CLI, Jobs API, or SDK to create the job:

```bash
nemo safe-synthesizer generate \
  --workspace default \
  --spec-file nss-job.json
```

For SDK users, point them to the Python SDK builder documented in `docs/safe-synthesizer/tutorials/safe-synthesizer-101.mdx`.

## Minimal Spec Shape

```json
{
  "data_source": "default/my-input#input.csv",
  "config": {
    "enable_synthesis": true,
    "enable_replace_pii": false,
    "generation": {
      "num_records": 100
    },
    "evaluation": {
      "enabled": true
    },
    "privacy": {
      "dp_enabled": false
    }
  }
}
```

For platform submission, pass this object as the `spec` field in the Jobs API or SDK create payload.

`platform-job.json` wraps the job spec:

```json
{
  "spec": {
    "data_source": "default/my-input#input.csv",
    "config": {
      "enable_synthesis": true,
      "enable_replace_pii": false
    }
  }
}
```

## Next Steps

- Tune job parameters with `workflows/config.md` and `workflows/config-runs.md`.
- Reuse a prior adapter with `pretrained_model_job`.
- Retrieve job result files with `workflows/results.md`.
- Interpret output files with `workflows/artifacts.md`.
- Debug failed runs with `workflows/diagnose.md`.
