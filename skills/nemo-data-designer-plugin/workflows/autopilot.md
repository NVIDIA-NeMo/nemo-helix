# Autopilot Workflow

In this mode, make reasonable design decisions autonomously based on the dataset description. Do not ask clarifying questions — infer sensible defaults and move straight through to a working preview.

1. **Resolve CLI command** — Run `command -v nemo 2>/dev/null || (test -x .venv/bin/nemo && realpath .venv/bin/nemo) || echo CLI_NOT_FOUND`.
  - If the output is a path, use it in place of `nemo` in every `nemo …` invocation in this workflow — including commands outside the `data-designer` group, such as `<path> inference providers list`.
  - If the output is `CLI_NOT_FOUND`, STOP and follow the Troubleshooting section in SKILL.md. Do not continue to the next step.
2. **Learn** — Run `nemo data-designer agent context`.
  - Read schemas for every column, sampler type, validator, and processor you plan to use, from the `config_root` path it prints.
  - Never guess types or parameters — read the relevant config files first.
  - Always read `base.py` for inherited fields shared by all config objects.
  - Ignore its **Model Aliases**, **Persona Datasets**, and **Commands** sections. Those describe a standalone local install; this skill runs on NeMo Platform, where aliases are declared in the script, persona data lives in platform filesets, and every command is prefixed with `nemo`. Read `references/platform-execution.md` instead.
  - Run `nemo inference providers list` to pick a provider for any LLM columns. Note that choice as one of the key decisions in step 3.
3. **Infer** — Based on the dataset description, make reasonable decisions for:
  - Axes of diversity and what should be well represented.
  - Which variables to randomize.
  - The schema of the final dataset.
  - The structure of any structured output columns.
  - Briefly state the key decisions you made so the user can course-correct if needed.
4. **Plan** — Determine columns, samplers, processors, validators, and other dataset features needed.
5. **Build** — Write the Python script with `load_config_builder()` returning a `DataDesignerConfigBuilder` (see Output Template in SKILL.md).
6. **Validate** — Run `nemo data-designer validate <path>`. Address any warnings or errors and re-validate until it passes.
   - `validate` checks the config's structure and resolves the platform resources it names. It does not check that the models respond.
   - Once it passes, run `nemo data-designer check-models <path>` once before the first preview. This catches a model name the provider cannot actually serve, which `validate` cannot detect. Re-run it only after changing a model or provider — each run bills a real generation per model alias.
7. **Preview** — Run `nemo data-designer preview <path> --save-results` to generate sample records as HTML files.
  - Note the sample records directory printed by the `nemo data-designer preview` command
  - Give the user a clickable link: `file://<sample-records-dir>/sample_records_browser.html`
8. **Create** — If the user specified a record count:
  - Run `nemo data-designer create <path> --num-records <N>`.
  - Generation speed depends heavily on the dataset configuration and the user's inference setup. For larger datasets, warn the user and ask for confirmation before running.
  - If no record count was specified, skip this step.
9. **Present** — Summarize what was built: columns, samplers used, key design choices. If the create command was run, share the results. Ask the user if they want any changes. If so, edit the script, re-validate, re-preview, and iterate.
