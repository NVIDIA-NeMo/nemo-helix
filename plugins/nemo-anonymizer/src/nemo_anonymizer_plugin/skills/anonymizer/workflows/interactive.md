<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Interactive Workflow

This is an interactive, iterative anonymization design process. Do not disengage from the loop unless the user says they are satisfied.

Source of truth for this workflow: `docs/anonymizer/tutorials/index.mdx`, `docs/anonymizer/tutorials/preview.mdx`, and `docs/anonymizer/tutorials/run.mdx`. Defer to them if the CLI flags or capabilities here look out of date.

1. **Resolve CLI command** — Run `command -v nemo 2>/dev/null || (test -x .venv/bin/nemo && realpath .venv/bin/nemo) || echo CLI_NOT_FOUND`.
   - If the output is a path, make sure that directory is on `PATH`, then use plain `nemo ...` commands below.
   - If the output is `CLI_NOT_FOUND`, STOP and follow the Troubleshooting section in SKILL.md. Do not continue.
2. **Confirm the plugin service is mounted before previewing or running.** Run `nhx_base_url="${NHX_BASE_URL:-http://localhost:8080}"; curl -sf "${nhx_base_url%/}/apis/anonymizer/v2/workspaces/${NHX_WORKSPACE:-default}/entity-labels" | jq -r '.data[0] // empty'`. If nothing prints, the plugin service isn't loaded — `nemo setup` does not auto-mount it. Tell the user to run `nemo services run` (no `--services` flag) and rerun the check.
3. **Confirm input source** — Decide whether the input is an `http(s)://` URL or a NeMo Helix fileset reference. If the user named a local file, ask whether to upload it to a fileset or host it first (see `references/inputs.md`).
4. **Clarify** — Ask the user clarifying questions to narrow down precisely what they want. Prefer a structured question tool if one is available, batch related questions together, keep the set short, and offer concrete options/defaults. Common things to make precise:
   - **Text column** to scan and (optional) **id column**.
   - **What to do with detected entities**: redact, annotate (tag inline), hash (deterministic token), substitute with realistic LLM-generated values, or fully rewrite the text under a privacy goal. See `references/replace-strategies.md` and `references/rewrite-mode.md`.
   - **Detection tuning** — keep Anonymizer library defaults unless the user explicitly asks for label/threshold changes; refer to the [Anonymizer library docs](https://github.com/NVIDIA-NeMo/Anonymizer/tree/main/docs) or library skills for those details.
   - **Preview surface** — `nemo anonymizer preview` through the plugin service, or `sdk.anonymizer.preview` if the user wants Python.
   - **Run surface** — `nemo anonymizer run` for Jobs-worker execution.
5. **Resolve model providers** — Ask which provider(s) and model aliases to use, or default to the aliases in `references/model-configs.md`. For provider discovery or creation, refer to the platform inference/model-provider docs or the relevant inference/model skill.
6. **Plan** — Summarize the planned config (replace vs rewrite strategy, detection tuning, model_configs, input source, num_records, preview surface) and ask the user to confirm before writing the spec.
7. **Build** — Write a YAML spec file following the Output Template in SKILL.md. Use the **Preview** shape first.
8. **Validate (optional)** — If you've also produced a stand-alone `AnonymizerConfig` YAML (e.g., the user wants `nemo anonymizer validate` to gate the run), invoke it now and address any errors before previewing.
9. **Preview** — Run `nemo anonymizer preview --spec-file <path> --workspace <ws>` with the preview spec, unless the user chose the SDK.

   Inspect the resulting frames: `log` lines, the `preview_dataset`, the `trace_dataset`, and any `failed_records`. Surface anything in `failed_records` to the user.
10. **Iterate**
    - Ask the user for feedback on the preview output. Offer to review the records yourself and suggest plugin-surface fixes (input source, model configs, selected model aliases) or refer to Anonymizer library docs/skills for library-level tuning. See `references/preview-review.md`.
    - Apply changes, re-preview. Repeat until the user is satisfied.
11. **Finalize** — Once the user is happy with the preview:
    - Generate a run spec by dropping `num_records` from the preview request. Run writes artifacts, not a dataset entity.
    - Tell the user they can run the full job with:

      ```bash
      nemo anonymizer run --spec-file <run_spec>.yaml --workspace <ws>
      ```

    - For jobs, show the CLI follow-up commands:

      ```bash
      nemo jobs get-status <job-name> --workspace <ws>
      nemo jobs get-logs <job-name> --workspace <ws> --all-pages
      nemo jobs results list <job-name> --workspace <ws>
      nemo jobs results download artifacts --job <job-name> --workspace <ws> --output-file artifacts.tar.gz
      ```
    - Caution that runtime depends on dataset size and the chosen strategy (LLM-backed strategies are slower).
    - Do not run the full job yourself — let the user decide when to launch it.
