# Interactive Workflow

This is an interactive, iterative design process. Do not disengage from the loop unless the user says they are satisfied.

1. **Resolve CLI command** — Run `command -v nemo 2>/dev/null || (test -x .venv/bin/nemo && realpath .venv/bin/nemo) || echo CLI_NOT_FOUND`.
  - If the output is a path, use `<path> data-designer` as the command prefix for all `nemo data-designer …` invocations in this workflow.
  - If the output is `CLI_NOT_FOUND`, STOP and follow the Troubleshooting section in SKILL.md. Do not continue to the next step.
2. **Learn** — Run `nemo data-designer agent context`.
  - Read schemas for every column, sampler type, validator, and processor you plan to use, from the `config_root` path it prints.
  - Never guess types or parameters — read the relevant config files first.
  - Always read `base.py` for inherited fields shared by all config objects.
  - Ignore its **Model Aliases**, **Persona Datasets**, and **Commands** sections. Those describe a standalone local install; this skill runs on NeMo Platform, where aliases are declared in the script, persona data lives in platform filesets, and every command is prefixed with `nemo`. Read `references/platform-execution.md` instead.
3. **Clarify** — Ask the user clarifying questions to narrow down precisely what they want.
  - Optimize for a great user experience: prefer a structured question tool over plain text if one is available, batch related questions together, keep the set short, provide concrete options/examples/defaults where possible, and use structured inputs (single-select, multi-select, free text, etc.) when they make answering easier.
  - If the dataset uses LLM columns, confirm with the user which provider/model(s) to use. Run `nemo inference providers list` to see what is registered, and offer those as options. See `references/platform-execution.md`.
  - Common things to make precise:
    - What the "axes of diversity" are — what should be well represented and diverse in the resulting dataset.
    - The kind and nature of any input data.
    - What variables should be randomized.
    - The schema of the final dataset.
    - The structure of any required structured output columns.
    - What facets of the output dataset are important to capture.
4. **Plan** — Determine columns, samplers, processors, validators, and other dataset features needed. Present the plan to the user and ask if they want any changes before generating a preview.
5. **Build** — Write the Python script with `load_config_builder()` returning a `DataDesignerConfigBuilder` (see Output Template in SKILL.md).
6. **Validate** — Run `nemo data-designer validate <path>`. Address any warnings or errors and re-validate until it passes.
7. **Preview** — Run `nemo data-designer preview <path> --save-results` to generate sample records as HTML files.
  - Note the sample records directory printed by the `nemo data-designer preview` command
  - Give the user a clickable link: `file://<sample-records-dir>/sample_records_browser.html`
8. **Iterate**
   - Ask the user for feedback.
   - Offer to review the records yourself and suggest improvements. If the user accepts, read `references/preview-review.md` for guidance.
   - Apply changes, re-validate, and re-preview. Repeat until the user is satisfied.
9. **Finalize** — Once the user is happy, tell them they can run the following command to create the dataset:
  - `nemo data-designer create <path> --num-records <N>`.
  - Caution the user that generation speed depends heavily on the dataset configuration and their inference setup.
  - Do not run this command yourself — the user should control when it runs.
