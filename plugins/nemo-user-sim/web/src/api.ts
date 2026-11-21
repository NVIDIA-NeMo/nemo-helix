// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobTerminalStatuses, JOB_POLLING_INTERVAL_MS } from '@nemo/common';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';
import { useHost } from './host';
import type {
  Job,
  JobCollection,
  PersonaPreview,
  Probe,
  ProbeSuggestions,
  ScenarioSet,
  Scores,
  SimulateRequest,
  Summary,
  Transcripts,
} from './types';

class ApiError extends Error {
  constructor(
    readonly status: number,
    detail: string
  ) {
    super(`${status}: ${detail}`);
  }
}

export const isFinished = (status?: string): boolean =>
  HelixJobTerminalStatuses.some((terminal) => terminal === status);

const jobsPath = (workspace: string, collection: JobCollection): string =>
  `/apis/user-sim/v2/workspaces/${encodeURIComponent(workspace)}/jobs/${collection}`;

// Studio's SDK covers platform services only, so the plugin calls its own API
// with the host's bearer, read per request so silent renew keeps working.
const useRequest = () => {
  const { apiBaseUrl, auth } = useHost();
  return useCallback(
    async <T>(path: string, init?: RequestInit): Promise<T> => {
      const token = auth.getAccessToken();
      const response = await fetch(`${apiBaseUrl}${path}`, {
        ...init,
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      });
      if (!response.ok) throw new ApiError(response.status, await response.text());
      // DELETE answers with no body.
      return (response.status === 204 ? undefined : await response.json()) as T;
    },
    [apiBaseUrl, auth]
  );
};

/** The probes a run can use, from the plugin's catalog. */
export const useProbes = () => {
  const request = useRequest();
  return useQuery({
    queryKey: ['user-sim', 'probes'],
    queryFn: () => request<Probe[]>('/apis/user-sim/v1/probes'),
    staleTime: Infinity,
  });
};

export const useSimulateJobs = () => {
  const { workspaceId } = useHost();
  const request = useRequest();
  return useQuery({
    queryKey: ['user-sim', workspaceId, 'simulate'],
    queryFn: async () => {
      const page = await request<{ data: Job[] }>(
        `${jobsPath(workspaceId, 'simulate')}?page_size=100`
      );
      return [...page.data].sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''));
    },
    refetchInterval: (query) =>
      query.state.data?.some((job) => !isFinished(job.status)) ? JOB_POLLING_INTERVAL_MS : false,
  });
};

export const useSimulateJob = (name: string) => {
  const { workspaceId } = useHost();
  const request = useRequest();
  return useQuery({
    queryKey: ['user-sim', workspaceId, 'simulate', name],
    queryFn: () => request<Job>(`${jobsPath(workspaceId, 'simulate')}/${encodeURIComponent(name)}`),
    refetchInterval: (query) =>
      isFinished(query.state.data?.status) ? false : JOB_POLLING_INTERVAL_MS,
  });
};

/**
 * A simulate job's JSON result. The job rewrites `summary` and `transcripts` as conversations finish, so
 * results are polled until the run ends. A missing result is null: `scores` only exists once evaluation ran.
 */
const useResult = <T>(
  job: string,
  result: 'summary' | 'scores' | 'transcripts' | 'scoring_error',
  finished: boolean
) => {
  const { workspaceId } = useHost();
  const request = useRequest();
  return useQuery({
    // Keyed on `finished` so the final version is fetched once when the run ends.
    queryKey: ['user-sim', workspaceId, 'simulate', job, result, finished],
    queryFn: () =>
      request<T>(
        `${jobsPath(workspaceId, 'simulate')}/${encodeURIComponent(job)}/results/${result}/download`
      ).catch((error: unknown) => {
        if (error instanceof ApiError && error.status === 404) return null;
        throw error;
      }),
    refetchInterval: finished ? false : JOB_POLLING_INTERVAL_MS,
    staleTime: finished ? Infinity : 0,
  });
};

export const useRunResults = (job: string, finished: boolean) => ({
  summary: useResult<Summary>(job, 'summary', finished),
  scores: useResult<Scores>(job, 'scores', finished),
  transcripts: useResult<Transcripts>(job, 'transcripts', finished),
  // Saved only when the judge couldn't score; says why in place of the job's bare exit code.
  scoringError: useResult<{ message: string }>(job, 'scoring_error', finished),
});

export const useCreateJob = <Spec>(collection: JobCollection) => {
  const { workspaceId } = useHost();
  const request = useRequest();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (spec: Spec) =>
      request<Job<Spec>>(jobsPath(workspaceId, collection), {
        method: 'POST',
        body: JSON.stringify({ spec }),
      }),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ['user-sim', workspaceId, collection] }),
  });
};

/** Run ``action`` on each named simulate run, then refresh the run list. Settles every run before failing. */
const useRunsAction = (method: 'DELETE' | 'POST', suffix: string) => {
  const { workspaceId } = useHost();
  const request = useRequest();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (names: string[]) => {
      const results = await Promise.allSettled(
        names.map((name) =>
          request(`${jobsPath(workspaceId, 'simulate')}/${encodeURIComponent(name)}${suffix}`, {
            method,
          })
        )
      );
      const failed = names.filter((_, index) => results[index].status === 'rejected');
      if (failed.length)
        throw new Error(`${failed.length} of ${names.length} failed: ${failed.join(', ')}`);
    },
    onSettled: () =>
      queryClient.invalidateQueries({ queryKey: ['user-sim', workspaceId, 'simulate'] }),
  });
};

export const useDeleteRuns = () => useRunsAction('DELETE', '');

export const useCancelRuns = () => useRunsAction('POST', '/cancel');

/** Registered agents in the workspace, for picking what to test. */
export const useAgents = () => {
  const { workspaceId } = useHost();
  const request = useRequest();
  return useQuery({
    queryKey: ['user-sim', workspaceId, 'agents'],
    queryFn: async () => {
      const page = await request<{ data: { name: string; description?: string }[] }>(
        `/apis/agents/v2/workspaces/${encodeURIComponent(workspaceId)}/agents?page_size=100`
      );
      return page.data;
    },
  });
};

/** Models the workspace's gateway serves, by the entity name a run takes. */
export const useModels = () => {
  const { workspaceId } = useHost();
  const request = useRequest();
  return useQuery({
    queryKey: ['user-sim', workspaceId, 'models'],
    queryFn: async () => {
      const page = await request<{ data: { id: string }[] }>(
        `/apis/inference-gateway/v2/workspaces/${encodeURIComponent(workspaceId)}/openai/-/v1/models`
      );
      return page.data.map((model) => model.id.split('/').slice(1).join('/') || model.id).sort();
    },
  });
};

/** Probes inferred from a registered agent's config and ETHOS.md. */
export const useProbeSuggestions = (agent: string | null) => {
  const { workspaceId } = useHost();
  const request = useRequest();
  return useQuery({
    queryKey: ['user-sim', workspaceId, 'probe-suggestions', agent],
    queryFn: () =>
      request<ProbeSuggestions>(
        `/apis/user-sim/v2/workspaces/${encodeURIComponent(workspaceId)}/probe-suggestions`,
        { method: 'POST', body: JSON.stringify({ agent }) }
      ),
    enabled: !!agent,
    staleTime: Infinity,
  });
};

/** Topic sets and tool-use request banks uploaded to the workspace with the CLI. */
export const useScenarioSets = () => {
  const { workspaceId } = useHost();
  const request = useRequest();
  return useQuery({
    queryKey: ['user-sim', workspaceId, 'scenario-sets'],
    queryFn: () =>
      request<ScenarioSet[]>(`/apis/user-sim/v2/workspaces/${encodeURIComponent(workspaceId)}/scenario-sets`),
  });
};

/** Pick the personas a run would meet; run from the returned preview to meet exactly them. */
export const usePersonaPreview = () => {
  const { workspaceId } = useHost();
  const request = useRequest();
  return useMutation({
    mutationFn: (spec: SimulateRequest) =>
      request<PersonaPreview>(
        `/apis/user-sim/v2/workspaces/${encodeURIComponent(workspaceId)}/persona-preview`,
        { method: 'POST', body: JSON.stringify(spec) }
      ),
  });
};

const annotationsPath = (workspace: string) =>
  `/apis/intake/v2/workspaces/${encodeURIComponent(workspace)}/annotations`;

/** A reviewer's latest thumbs on a conversation's Intake session: 'positive', 'negative' or null. */
export const useFeedback = (sessionId: string | null) => {
  const { workspaceId } = useHost();
  const request = useRequest();
  return useQuery({
    queryKey: ['user-sim', workspaceId, 'feedback', sessionId],
    queryFn: async () => {
      const filter = encodeURIComponent(
        JSON.stringify({ session_id: sessionId, kind: 'feedback' })
      );
      const page = await request<{ data: { value: string }[] }>(
        `${annotationsPath(workspaceId)}?filter=${filter}&sort=-created_at&page_size=1`
      );
      return page.data[0]?.value ?? null;
    },
    enabled: !!sessionId,
  });
};

/** Record a thumbs or a note on a conversation's Intake session, as Studio's trace view does. */
export const useAnnotate = () => {
  const { workspaceId } = useHost();
  const request = useRequest();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (
      annotation:
        | { session_id: string; kind: 'feedback'; value: 'positive' | 'negative' }
        | { session_id: string; kind: 'note'; text: string }
    ) =>
      request(annotationsPath(workspaceId), { method: 'POST', body: JSON.stringify(annotation) }),
    onSuccess: (_, annotation) =>
      queryClient.invalidateQueries({
        queryKey: ['user-sim', workspaceId, 'feedback', annotation.session_id],
      }),
  });
};
