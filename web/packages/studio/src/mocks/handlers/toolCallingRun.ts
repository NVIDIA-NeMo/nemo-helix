// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import type { ToolCallingResult } from '@studio/routes/agents/AgentOptimizationDetailRoute/toolCalling/toolCallingResults';
import { http, HttpResponse } from 'msw';

const OPTIMIZE_JOB_URL = `${PLATFORM_BASE_URL}/apis/agent-optimization/v2/workspaces/:workspace/jobs/run-strategy/:name`;
const FILESETS_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`;
const RESULTS_FILESET = 'tool-calling-artifacts';
const RESULTS_DIR = 'results/tool-calling';

const metrics = (
  rejected: number,
  toolCalls: number,
  firstCall: number,
  success: number,
  runsWithErrors: number,
  signals: [toolError: number, schemaInvalid: number, errorText: number, retryLoopRuns: number]
) => ({
  signals: {
    tool_error: signals[0],
    schema_invalid: signals[1],
    error_text: signals[2],
    retry_loop_runs: signals[3],
  },
  runs: 15,
  runs_with_errors: runsWithErrors,
  tool_calls: toolCalls,
  rejected_calls: rejected,
  tool_error_rate: Math.round((rejected / toolCalls) * 1000) / 1000,
  first_call_accepted_rate: firstCall,
  task_success_rate: success,
  avg_calls_per_run: Math.round((toolCalls / 15) * 10) / 10,
});

export const TOOL_CALLING_RESULT_FIXTURE: ToolCallingResult = {
  version: 1,
  agent: 'default/data-designer-toolcall',
  analyzer_model: 'default/nemotron-3-5-super',
  dataset: { prompts: 5, runs_per_prompt: 3 },
  variants: [
    {
      id: 'baseline',
      label: 'Current agent',
      description: 'The agent exactly as registered, with no changes.',
      metrics: metrics(48, 60, 0.13, 0.4, 1, [45, 40, 2, 9]),
      rejection_reasons: { 'corrupted or unknown key': 22, 'model/provider not on platform': 12 },
      examples: [
        {
          prompt_id: 'product-reviews',
          tool: 'generate_data_designer_job_request',
          arguments_excerpt: '{"job_request": {"columns": [{"sampler_type ": "category"}]}}',
          rejection: "columns[0]: unknown field 'sampler_type '",
          signals: ['tool_error', 'schema_invalid'],
          schema_errors: [
            "job_request/spec: Additional properties are not allowed ('model_configs' was unexpected)",
          ],
        },
      ],
    },
    {
      id: 'config-fix',
      label: 'Prompt fix',
      description: 'The current agent with the suggested change to its prompt and config.',
      metrics: metrics(14, 30, 0.47, 0.73, 0, [13, 12, 1, 4]),
      rejection_reasons: { 'corrupted or unknown key': 14 },
      examples: [],
    },
    {
      id: 'schema-fix',
      label: 'Schema fix',
      description:
        "The prompt fix, plus the suggested tool schema served in place of the tool's own.",
      metrics: metrics(1, 17, 0.93, 1, 0, [1, 0, 0, 0]),
      rejection_reasons: { 'corrupted or unknown key': 1 },
      examples: [],
    },
  ],
  findings: [
    {
      id: 'lint-open-object-columns',
      source: 'schema-lint',
      severity: 'high',
      tool: 'generate_data_designer_job_request',
      title: 'Column items allow any key',
      detail: 'additionalProperties: true on job_request.spec.config.columns[]; 12 optional keys.',
      evidence: ["22 calls rejected for unknown key 'sampler_type '"],
    },
    {
      id: 'failure-unknown-provider',
      source: 'failure-analysis',
      severity: 'medium',
      tool: null,
      title: 'Agent names providers the platform does not serve',
      detail: "The prompt never names a model, so the agent guesses 'openai'.",
      evidence: [],
    },
  ],
  suggestions: [
    {
      id: 'config-fix',
      kind: 'agent-config',
      title: "Name the platform's only model in the prompt",
      rationale: "12 rejections used provider 'openai'.",
      applies_automatically: true,
      artifact: 'candidate-agent.yaml',
      diff_artifact: 'candidate-agent.diff',
      verified_by_variant: 'config-fix',
    },
    {
      id: 'schema-fix',
      kind: 'tool-schema',
      title: 'One schema per column type, required keys first',
      rationale: 'Closed schemas stop the model inventing keys.',
      applies_automatically: false,
      artifact: 'schema-patch/data_designer/generate_data_designer_job_request.json',
      diff_artifact: 'schema-patch.diff',
      verified_by_variant: 'schema-fix',
    },
  ],
  recommended_variant: 'config-fix',
  recommendation:
    "Apply the prompt fix now: it cuts the tool error rate from 80% to 47%. The schema patch would help further but needs the tool's owner.",
};

export const CANDIDATE_AGENT_DIFF = `--- agent.yaml
+++ candidate-agent.yaml
@@ -3,2 +3,2 @@
-  system_prompt: Build a Data Designer job.
+  system_prompt: Build a Data Designer job with model default/nemotron-3-5-super.`;

export const SCHEMA_PATCH_DIFF = `--- generate_data_designer_job_request.json
+++ generate_data_designer_job_request.json
@@ -1,1 +1,1 @@
-  "additionalProperties": true
+  "additionalProperties": false`;

interface ToolCallingRunOptions {
  workspace: string;
  jobName: string;
  status?: HelixJobStatus;
  strategy?: string;
  /** Body of `tool-calling-result.json`; `null` leaves the file out of the results. */
  resultBody?: string | null;
}

/** A finished `tool-calling` optimize job whose results fileset holds the contract's files. */
export const toolCallingRunHandlers = ({
  workspace,
  jobName,
  status = 'completed',
  strategy = 'tool-calling',
  resultBody = JSON.stringify(TOOL_CALLING_RESULT_FIXTURE),
}: ToolCallingRunOptions) => {
  const files: Record<string, string> = {
    'candidate-agent.diff': CANDIDATE_AGENT_DIFF,
    'schema-patch.diff': SCHEMA_PATCH_DIFF,
    'candidate-agent.yaml': 'name: data-designer-toolcall\n',
    ...(resultBody === null ? {} : { 'tool-calling-result.json': resultBody }),
  };

  return [
    http.get(OPTIMIZE_JOB_URL, () =>
      HttpResponse.json({
        name: jobName,
        workspace,
        status,
        created_at: '2026-11-18T09:00:00Z',
        updated_at: '2026-11-18T09:20:00Z',
        spec: { strategy, agent: 'data-designer-toolcall' },
      })
    ),
    http.get(`${OPTIMIZE_JOB_URL}/status`, () =>
      HttpResponse.json({
        id: 'opt-tool-calling',
        name: jobName,
        status,
        status_details: {},
        error_details: {},
        steps: [],
        created_at: '2026-11-18T09:00:00Z',
        updated_at: '2026-11-18T09:20:00Z',
      })
    ),
    http.get(`${OPTIMIZE_JOB_URL}/results`, () =>
      HttpResponse.json({
        data: [
          {
            name: 'tool-calling',
            artifact_storage_type: 'fileset',
            artifact_url: `fileset://${workspace}/${RESULTS_FILESET}#${RESULTS_DIR}`,
          },
        ],
      })
    ),
    http.get(`${FILESETS_URL}/${RESULTS_FILESET}/files`, () =>
      HttpResponse.json({
        data: Object.keys(files).map((name) => ({
          path: `${RESULTS_DIR}/${name}`,
          size: files[name].length,
          file_ref: name,
        })),
      })
    ),
    http.get(`${FILESETS_URL}/${RESULTS_FILESET}/-/*`, ({ request }) => {
      const path = decodeURIComponent(new URL(request.url).pathname.split('/-/')[1] ?? '');
      const name = path.startsWith(`${RESULTS_DIR}/`) ? path.slice(RESULTS_DIR.length + 1) : '';
      const body = files[name];
      if (body === undefined) {
        return new HttpResponse(null, { status: 404 });
      }
      return new HttpResponse(body);
    }),
  ];
};
