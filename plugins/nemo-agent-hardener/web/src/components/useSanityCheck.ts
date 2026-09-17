// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { customFetch } from '@agent-hardener/api/fetcher';
import { parseJson, parseText, useJobArtifact } from '@agent-hardener/components/useJobArtifact';
import type { Mitigations } from '@agent-hardener/components/useMitigations';
import { agentHardenerListRuns, useAgentHardenerCreateJob } from '@agent-hardener/generated/api';
import type { AgentHardenerRun, PlatformJobStatus } from '@agent-hardener/generated/schema';
import { JOB_POLLING_INTERVAL_MS } from '@nemo/common';
import { useMutation, useQuery } from '@tanstack/react-query';

// The validate-only run writes a `validation` job result: per-item attack/benign verdicts for the scorecard.
const VALIDATION_RESULT = 'validation';
// …and a `composed-guardrails` result: the exact plugins.toml it validated, so "Apply to Agent" survives a reload.
const COMPOSED_GUARDRAILS_RESULT = 'composed-guardrails';

export interface ValidationAttackRow {
  attack_id?: string;
  probe?: string;
  goal?: string;
  prompt_excerpt?: string;
  status?: 'blocked' | 'not_blocked' | 'error';
  confidence?: number;
}

export interface ValidationBenignRow {
  index?: number;
  tool?: string;
  label?: string;
  persona?: string;
  payload_excerpt?: string;
  status?: 'passed' | 'refused' | 'error';
  confidence?: number;
}

export interface ValidationSummary {
  attacks_total: number;
  attacks_blocked: number;
  benign_total: number;
  benign_false_positives: number;
}

export interface ValidationReport {
  attacks: ValidationAttackRow[];
  benign: ValidationBenignRow[];
  summary: ValidationSummary;
}

interface ComposeDefenseResponse {
  guardrails_toml?: string | null;
  policy_yaml?: string | null;
}

// Compose the chosen subset of a run's defenses into a deployable plugins.toml + policy YAML (server-side, so the
// same composition feeds the live preview and the sanity-check run). Sends the mitigations the client already
// holds plus the selected ids.
export const useComposeDefense = (workspace: string, runName: string) =>
  useMutation({
    mutationFn: (vars: {
      mitigations: Mitigations;
      selectedDefenseIds: string[];
    }): Promise<ComposeDefenseResponse> =>
      customFetch<ComposeDefenseResponse>({
        url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(workspace)}/runs/${encodeURIComponent(runName)}/compose-defense`,
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        data: { mitigations: vars.mitigations, selected_defense_ids: vars.selectedDefenseIds },
      }),
  });

interface SanityCheckSpec {
  manifest_id: string;
  driver: 'service';
  validate_only: true;
  replay_hitlog_fileset: string;
  defense_guardrails?: string;
  defense_policy?: string;
  // The harden run this check was launched from — recorded on the sanity run so the report re-attaches on reload.
  source_run: string;
}

// Submit a validate-only war-game: freeze the composed defenses as the victim baseline and replay the
// recorded attacks + benign suite against it. Returns the created job name so the caller can poll its result.
export const useSubmitSanityCheck = (workspace: string) => {
  const createJob = useAgentHardenerCreateJob();
  const submit = async (spec: SanityCheckSpec): Promise<string> => {
    // The backend WarGameSpec accepts validate_only/defense_* (added for the sanity check); the generated
    // client type predates them, so pass the spec through the untyped job request field.
    const job = await createJob.mutateAsync({ workspace, data: { spec } as never });
    return job.name;
  };
  return { submit, isPending: createJob.isPending };
};

// Poll the sanity-check job's results until `validation.json` lands, then download + parse it once.
// `status` stops the poll on a failed/cancelled job, which would otherwise never write the result.
export const useSanityCheckResult = (
  workspace: string,
  jobName: string | undefined,
  status?: PlatformJobStatus
) => {
  const artifact = useJobArtifact<ValidationReport>(
    workspace,
    jobName,
    VALIDATION_RESULT,
    parseJson,
    status
  );
  return {
    report: artifact.data,
    isLoading: artifact.isLoading,
    hasReport: artifact.present,
    missingReport: artifact.missing,
  };
};

// Recover the guardrail set a sanity check validated (its `composed-guardrails` result), so "Apply to Agent"
// stays enabled after a reload when the in-memory copy is gone.
export const useSanityCheckComposedGuardrails = (
  workspace: string,
  jobName: string | undefined,
  status?: PlatformJobStatus
): string | undefined =>
  useJobArtifact(workspace, jobName, COMPOSED_GUARDRAILS_RESULT, parseText, status).data;

// The run entity carries `source_run` (set on validate-only sanity checks); the generated type predates it.
type RunWithSource = AgentHardenerRun & { source_run?: string; job_id?: string; name?: string };

// Find the job of the most recent sanity check launched from `runName`, so the Harden tab can re-attach the
// scorecard after a reload / re-visit (the in-memory job name doesn't survive a full page load). Sanity runs
// are the only runs with `source_run` set, so matching it uniquely identifies this run's checks.
export const useLatestSanityCheckJob = (workspace: string, runName: string): string | undefined => {
  const { data } = useQuery({
    queryKey: ['agent-hardener-sanity-lookup', workspace, runName],
    enabled: Boolean(runName),
    // Stop once a job is found; re-attaching only needs the lookup until then. Polling forever kept
    // a request every 5s running for the whole time the Harden tab stayed open.
    refetchInterval: (query) => (query.state.data ? false : JOB_POLLING_INTERVAL_MS),
    queryFn: async (): Promise<string | undefined> => {
      const res = await agentHardenerListRuns(workspace, { sort: '-created_at', page_size: 50 });
      const runs = (res.data ?? []) as RunWithSource[];
      // Newest-first, so the first match is the latest sanity check for this run.
      return runs.find((r) => r.source_run === runName)?.job_id;
    },
  });
  return data ?? undefined;
};
