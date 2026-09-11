// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  filesCreateFileset,
  filesListFilesetFiles,
  filesUploadFile,
} from '@nemo/sdk/generated/platform/files';

// Idempotent seeding of an eval-config fileset, shared by the agent-evaluation
// submit flow (which supplies its own `eval-config.json`) and the optimizer
// apply flow (which relies on the bundled react-eval default below).

const isNotFoundError = (err: unknown): boolean => {
  const e = err as { response?: { status?: number }; status?: number };
  return e?.response?.status === 404 || e?.status === 404;
};

export const isConflictError = (err: unknown): boolean => {
  const e = err as { response?: { status?: number }; status?: number };
  return e?.response?.status === 409 || e?.status === 409;
};

const isCanceledError = (err: unknown): boolean => {
  const e = err as { name?: string; code?: string };
  return e?.name === 'AbortError' || e?.name === 'CanceledError' || e?.code === 'ERR_CANCELED';
};

/** Fileset name for an agent's eval bundle. */
export const evalFilesetForAgent = (agentName: string): string => `${agentName}-eval`;

export const SAMPLE_EVAL_CONFIG_PATH = 'react-eval.yml';
export const SAMPLE_EVAL_DATA_PATH = 'react-eval-data.json';

export const SAMPLE_EVAL_YAML = `# react-eval.yml — bundled sample seeded by the optimizer apply flow.
#
# Evaluates against the deployed agent endpoint. The judge LLM scores answers
# and must be available in the workspace.

llms:
  llm:
    _type: openai
    model_name: nvidia-nemotron-3-nano-30b-a3b
    temperature: 0.0
    max_tokens: 1024

  judge_llm:
    _type: openai
    model_name: nvidia-nemotron-3-super-120b-a12b
    temperature: 0.0
    max_tokens: 1024

eval:
  general:
    max_concurrency: 4
    output_dir: eval/agent
    dataset:
      _type: json
      file_path: ${SAMPLE_EVAL_DATA_PATH}
  evaluators:
    accuracy:
      _type: tunable_rag_evaluator
      llm_name: judge_llm
      default_scoring: true
      default_score_weights:
        coverage: 0.5
        correctness: 0.3
        relevance: 0.2
      judge_llm_prompt: >
        You are an evaluator. Score whether the generated answer correctly
        addresses the question compared to the expected answer description.
        Rules:
        - Score is a float between 0.0 and 1.0.
        - 1.0 means the answer fully satisfies the expected answer criteria.
        - Provide a 1-2 sentence reasoning.
`;

export const SAMPLE_EVAL_DATA_JSON = JSON.stringify(
  [
    {
      id: 1,
      question: 'Who invented the telephone, and what is the current time?',
      answer:
        'Answer must mention Alexander Graham Bell as the inventor of the telephone and include the current time',
    },
    {
      id: 2,
      question: 'What is the capital of France, and what day of the week is it today?',
      answer:
        'Answer must state that the capital of France is Paris and include the current day of the week',
    },
    {
      id: 3,
      question: "When was the theory of general relativity published, and what is today's date?",
      answer:
        "Answer must mention 1915 as the year general relativity was published and include today's date",
    },
  ],
  null,
  2
);

export interface EvalSeedFile {
  path: string;
  content: string;
  type: string;
}

/** Default seed files: the bundled react sample. Used by the optimizer apply
 *  flow and by the eval modal's fallback. */
const defaultEvalSeedFiles = (): EvalSeedFile[] => [
  { path: SAMPLE_EVAL_CONFIG_PATH, content: SAMPLE_EVAL_YAML, type: 'application/yaml' },
  { path: SAMPLE_EVAL_DATA_PATH, content: SAMPLE_EVAL_DATA_JSON, type: 'application/json' },
];

export const ensureEvalConfigFileset = async (
  workspace: string,
  fileset: string,
  signal: AbortSignal,
  files: EvalSeedFile[] = defaultEvalSeedFiles(),
  description?: string
): Promise<void> => {
  let existingPaths = new Set<string>();
  try {
    const listing = await filesListFilesetFiles(workspace, fileset, undefined, signal);
    existingPaths = new Set((listing?.data ?? []).map((f) => f.path));
  } catch (err) {
    if (isCanceledError(err)) throw err;
    if (!isNotFoundError(err)) throw err;
    try {
      await filesCreateFileset(workspace, { name: fileset, description }, signal);
    } catch (createErr) {
      if (isCanceledError(createErr)) throw createErr;
      if (!isConflictError(createErr)) throw createErr;
      const reListing = await filesListFilesetFiles(workspace, fileset, undefined, signal);
      existingPaths = new Set((reListing?.data ?? []).map((f) => f.path));
    }
  }
  const uploads = files.filter((f) => !existingPaths.has(f.path));
  for (const u of uploads) {
    const blob = new Blob([u.content], { type: u.type });
    await filesUploadFile(workspace, fileset, u.path, blob, signal);
  }
};
