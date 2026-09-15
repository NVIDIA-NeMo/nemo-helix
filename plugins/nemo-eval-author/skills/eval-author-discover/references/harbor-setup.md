<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Help the user get Harbor ready

Read this when discovery cannot use Harbor. The inventory script remains
read-only; it never installs dependencies. Missing Harbor and missing evals are
separate findings, and neither settles the other.

## Establish what is missing

First follow **Before you start** in discovery: inspect the repository's documented
environment and any existing Harbor launcher, then verify its interpreter. A
failed import in one environment does not prove Harbor is absent from the machine.
If a launcher exists but fails, report its actual error and investigate that
installation before suggesting another one. Do not upgrade or reinstall merely
because the current interpreter cannot use it.

## Explain the effect and the next step

After the usual Harbor introduction, explain the observed limitation briefly:

> I couldn't find a working Harbor installation in the environments I checked.
> We can still identify your evals, gather the requirements, and work through
> the grading rules. We'll need Harbor to create its task files, validate them,
> and run the evals. I'll keep our findings so we can pick up from here after setup.

Adapt this wording if an installation exists but is inaccessible or broken.
Keep the relevant checklist item open with the concrete setup requirement. Still
ask the source-selection and conversion questions at their normal points; do not
make installation a prerequisite for those conversations or restart them afterward.
After conversion is accepted, source mapping and grading design can continue
without Harbor. Do not fabricate a native task layout or call notes runnable tasks.

## Give supported installation instructions

Check the repository's setup instructions and Harbor version requirements first.
If it already manages Harbor, follow that documented environment rather than
introducing a competing installation. Otherwise, the stable installation in
[Harbor's getting-started guide](https://www.harborframework.com/docs/getting-started)
uses uv, a Python package and tool manager:

```bash
uv tool install harbor
harbor --help
```

Explain that the first command installs Harbor as a command-line tool; the second
checks that the command starts. Provide these as user-run instructions by default.
Do not automatically install or upgrade software as part of discovery or a broad
request to get evals working. If the user explicitly asks you to install Harbor,
use that authorization and the environment's normal permission mechanism, then
complete the verification below. Do not ask again for installation permission
already supplied. Installing Harbor does not authorize paid evaluation runs.

If uv is missing, link the [official uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
for the user's operating system. Offer guided setup if needed rather than dumping
multiple package-manager alternatives. If installation is blocked by network,
permissions, or a sandbox, explain the observed problem and how the user can run
the same documented setup in an appropriate terminal. Preserve the findings and
continue independent preparation; do not change repository dependencies to work
around a blocked tool installation.

## Verify and resume

After setup, check `harbor --help`, resolve the actual installed interpreter as
described in discovery, and verify it imports Harbor and reports its version.
For a uv tool installation, the project interpreter may still lack Harbor; reuse
the verified tool environment rather than repeating installation. Follow the
repository's documented environment if its agent imports need additional packages.

Rerun discovery with that verified interpreter and save the updated report. Resume
the existing source choice, accepted conversion, and gathered requirements.
Installing Harbor proves neither task readiness nor a successful agent run. Check
Docker or another execution backend, credentials, and application access later
according to the selected tasks; do not install those as part of this step.
