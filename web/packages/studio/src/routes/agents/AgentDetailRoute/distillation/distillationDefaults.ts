// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** The student model offered first when the workspace has one. */
export const PREFERRED_STUDENT_MODEL = /nemotron.*lightning/i;

/** What a distillation needs that the form does not ask for. */
export const DISTILLATION_DEFAULTS = {
  servingGpus: 1,
  trainingGpus: 1,
  gpuBudget: 1,
  candidateName: 'lora',
} as const;

interface ServingArgs {
  pattern: RegExp;
  args: string[];
}

/** vLLM tool-calling flags per model family. A model with no entry is served with the engine defaults. */
const SERVING_ARGS: ServingArgs[] = [
  {
    pattern: /nemotron.*lightning/i,
    args: ['--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'nemotron_v3'],
  },
  {
    pattern: /glm.*(5-3-flash|flash-5-3)/i,
    args: ['--enable-auto-tool-choice', '--tool-call-parser', 'glm47', '--reasoning-parser', 'glm47'],
  },
];

export const servingArgsFor = (model: string): string[] =>
  SERVING_ARGS.find(({ pattern }) => pattern.test(model))?.args ?? [];
