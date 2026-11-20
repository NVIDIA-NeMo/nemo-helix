<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Turning conversations into training data

`nemo user-sim export --job <job> --fileset <name>` keeps the conversations that passed scoring and writes them as chat JSONL (`{"messages": [...]}`) to a dataset fileset. The Helix post-training plugin reads that format directly.

## Quality gate

The `strict` gate (default) drops conversations with any judge concern. `generous` keeps borderline ones; pass it with `--spec '{"quality": "generous"}'`. Use strict for fine-tuning and generous only when the user needs volume and will review by hand.

## Human review in Studio

When the run published to Intake (the default), every conversation is a session in Studio. Reviewers open a conversation from the run page, then give the session a thumbs up or thumbs down.

- Without `--require-review`, export drops anything with a thumbs down and keeps the rest that passed.
- With `--require-review`, export keeps only conversations with a thumbs up.

Ask the user which they want before exporting. If nobody has reviewed yet, `--require-review` exports nothing.

## What gets dropped

Reasoning content is removed from assistant messages. Failed and timed-out conversations are never exported.

## Next step

Point the Helix post-training plugin at the fileset. Don't build a fine-tuning command from memory: use that plugin's own skill.
