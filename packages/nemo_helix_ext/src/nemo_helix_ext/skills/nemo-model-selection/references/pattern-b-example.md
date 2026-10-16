<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Pattern B — structure and example output

## Structure

1. State that there's a trade-off to settle before naming a model.
2. Present exactly two candidates labeled A and B (NOT named):
   - **A. The matched specialist** — the candidate that best fits the profile by name/intent, with its inferred/missing-data caveat.
   - **B. The measured generalist** — the model with the strongest *direct* score on the same primary axis, even if less specialized.
3. For each, summarize the `plain` text from the cache (which already carries the editorial caveat for inferred entries) in your own words.
4. Recommend a 5–10 prompt eval from the user's spec on both as the "real answer."
5. If the user can't run an eval, ask: which framing of your priority is right — A's specialist intent or B's measured discipline?
6. **Only after the user picks** — name the chosen model and proceed to Step 2's recommendation template (skipping the model-selection paragraph since it's resolved).

## Example

Example for a code-heavy agent where the natural pick has inferred BFCL data. Adapt the content; do not copy verbatim.

> "Before I name a model — one thing to settle. Your profile (code-focused, three tools, cloud) has two reasonable candidates with different evidence quality:
>
> **A. The specialist.** Built for this exact job. Arena's coding-category Elo (1418, strong) IS measured for this exact model. *But BFCL hasn't measured its tool-calling discipline — only its base model (Qwen3-30B-A3B at 37%) and we're inferring from there.*
>
> **B. The measured generalist.** Best directly-measured tool-calling in our NIM set (BFCL 52%, mid). Not code-specialized, but disciplined.
>
> If you can run a 5–10 prompt eval from your spec on both, that's the real answer. If you need to commit now: which matters more — specialist intent (A) or measured discipline (B)?"
