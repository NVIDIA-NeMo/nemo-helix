// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { usePlatformSdk } from '@agent-hardener/api/platform';
import {
  pendingInterview,
  pendingReview,
  type InterviewAnswer,
  type InterviewPrompt,
  type ReviewPrompt,
  type SuiteRow,
} from '@agent-hardener/components/hitlTypes';
import {
  currentActivity,
  reconSteps,
  type ReconStep,
} from '@agent-hardener/components/swarm/swarmModel';
import { useSwarmEvents } from '@agent-hardener/components/swarm/useSwarmEvents';
import {
  agentHardenerListRuns,
  useAgentHardenerCreateSynthBenignJob,
} from '@agent-hardener/generated/api';
import type { AgentHardenerRun } from '@agent-hardener/generated/schema';
import { useToast } from '@agent-hardener/host';
import { CJobTerminalStatuses, getJobRefetchInterval } from '@nemo/common';
import { useEffect, useMemo, useRef, useState } from 'react';

export interface GenerateBenignSuite {
  start: () => void;
  active: boolean; // a generate job is running or awaiting input
  starting: boolean; // job submitted, still resolving its run / first events
  status?: string;
  activity?: string; // live label for the current lifecycle stage (sandbox build, victim health, recon…)
  recon: ReconStep[];
  interview: InterviewPrompt | null;
  review: ReviewPrompt | null;
  submitInterview: (answers: InterviewAnswer[]) => void;
  submitReview: (suite: SuiteRow[]) => void;
  isResponding: boolean;
}

// Drive a manifest's benign-suite generation from the manifest page: launch the synth job with the
// service driver (serve HITL), track the run it creates (for the live recon SSE stream), and relay the
// interview/review HITL over the job's status_details. Unlike useRunWarGame this never navigates — the
// flow lives inline on the manifest.
export const useGenerateBenignSuite = (
  workspace: string,
  manifestName: string,
  onComplete?: () => void
): GenerateBenignSuite => {
  const toast = useToast();
  const [jobName, setJobName] = useState('');
  const [runName, setRunName] = useState('');
  const completedFor = useRef('');

  // The service-driven job creates its run record shortly after start (linked by job_id); poll for it so we
  // can subscribe to its event stream.
  const resolveRun = async (job: string): Promise<void> => {
    for (let attempt = 0; attempt < 15; attempt++) {
      const { data } = await agentHardenerListRuns(workspace, { sort: '-created_at', page_size: 20 });
      const run = (data as AgentHardenerRun[] | undefined)?.find((r) => r.job_id === job);
      if (run?.name) {
        setRunName(run.name);
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  };

  // Re-attach to a generation already in flight for this manifest.
  //
  // The HITL prompt lives on the job's status_details, but the job name lived only in this
  // component's state — so a reload, a navigation, or a second tab left the job waiting for an
  // answer with no UI asking for it. From the user's side the interview simply vanished, and the
  // job held its sandbox until the HITL timeout, which then collided with the next attempt.
  const adoptedFor = useRef('');
  useEffect(() => {
    if (jobName || adoptedFor.current === manifestName) return;
    adoptedFor.current = manifestName;
    void (async () => {
      const { data } = await agentHardenerListRuns(workspace, {
        sort: '-created_at',
        page_size: 1,
        filter: { manifest_id: manifestName, status: 'running' },
      });
      const run = (data as AgentHardenerRun[] | undefined)?.[0];
      // The job poll decides whether it is really still live; a stale run just resolves to terminal.
      if (run?.job_id) {
        setRunName(run.name ?? '');
        setJobName(run.job_id);
      }
    })();
  }, [workspace, manifestName, jobName]);

  const create = useAgentHardenerCreateSynthBenignJob({
    mutation: {
      onSuccess: (job) => {
        setRunName('');
        completedFor.current = '';
        adoptedFor.current = manifestName; // this session supersedes anything we might adopt
        setJobName(job.name);
        void resolveRun(job.name);
      },
      onError: () => toast.error('Failed to start benign-suite generation.'),
    },
  });

  const { useJobsGetJob, useJobsUpdateJobStatusDetails } = usePlatformSdk();
  const { data: job } = useJobsGetJob(workspace, jobName, {
    query: {
      enabled: Boolean(jobName),
      refetchInterval: (query) => getJobRefetchInterval(query.state.data?.status),
    },
  });
  const details = job?.status_details as Record<string, unknown> | undefined;
  const interview = pendingInterview(details);
  const review = pendingReview(details);

  const patch = useJobsUpdateJobStatusDetails();
  const respond = (data: Record<string, unknown>) =>
    patch.mutate({ workspace, name: jobName, data });

  const status = job?.status;
  const terminal = Boolean(status && CJobTerminalStatuses.includes(status));

  // Declared after `terminal` so the event poll can stop with the job; the synth run emits nothing once
  // its job reaches a terminal status.
  const events = useSwarmEvents(workspace, runName, terminal);
  const recon = useMemo(() => reconSteps(events), [events]);
  const activity = useMemo(() => currentActivity(events), [events]);

  useEffect(() => {
    if (jobName && terminal && completedFor.current !== jobName) {
      completedFor.current = jobName;
      if (status === 'completed') onComplete?.();
    }
  }, [jobName, terminal, status, onComplete]);

  return {
    start: () =>
      create.mutate({
        workspace,
        data: { spec: { manifest_id: manifestName, driver: 'service' } },
      }),
    active: Boolean(jobName) && !terminal,
    starting: Boolean(jobName) && !runName,
    status,
    activity,
    recon,
    interview,
    review,
    submitInterview: (answers) =>
      interview && respond({ interview_response: { round: interview.round, answers } }),
    submitReview: (suite) => review && respond({ review_response: { round: review.round, suite } }),
    isResponding: patch.isPending,
  };
};
