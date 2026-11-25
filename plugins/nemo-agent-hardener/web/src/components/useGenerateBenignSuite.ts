// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { toastApiError } from '@agent-hardener/api/apiError';
import type { JobFailure } from '@agent-hardener/api/jobFailure';
import { useHelixSdk, usePolledJob } from '@agent-hardener/api/platform';
import { waitForJobRun } from '@agent-hardener/api/runJobs';
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
import { useJobFailure } from '@agent-hardener/components/useJobFailure';
import {
  agentHardenerListRuns,
  useAgentHardenerCreateSynthBenignJob,
} from '@agent-hardener/generated/api';
import type { AgentHardenerRun } from '@agent-hardener/generated/schema';
import { useToast } from '@agent-hardener/host';
import { CJobTerminalStatuses } from '@nemo/common';
import { useEffect, useMemo, useRef, useState } from 'react';

export interface GenerateBenignSuite {
  start: () => void;
  active: boolean; // a generate job is being submitted, running, or awaiting input
  starting: boolean; // job submitted, still resolving its run / first events
  status?: string;
  jobName: string;
  failure?: JobFailure; // why the last generate job ended without a suite
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
    try {
      const run = await waitForJobRun(workspace, job, { attempts: 15, intervalMs: 1000 });
      if (run) setRunName(run);
    } catch (error) {
      toastApiError(toast, 'Could not follow the generation run')(error);
    }
  };

  // Re-attach to a generation already in flight for this manifest.
  //
  // The HITL prompt lives on the job's status_details, but the job name lived only in this
  // component's state — so a reload, a navigation, or a second tab left the job waiting for an
  // answer with no UI asking for it. From the user's side the interview simply vanished, and the
  // job held its sandbox until the HITL timeout, which then collided with the next attempt.
  const adoptedFor = useRef('');
  const [adopted, setAdopted] = useState(false);
  useEffect(() => {
    if (jobName || adoptedFor.current === manifestName) return;
    adoptedFor.current = manifestName;
    void (async () => {
      try {
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
          setAdopted(true);
        }
      } catch (error) {
        toastApiError(toast, 'Could not check for a generation in progress')(error);
      }
    })();
  }, [workspace, manifestName, jobName, toast]);

  const create = useAgentHardenerCreateSynthBenignJob({
    mutation: {
      onSuccess: (job) => {
        setRunName('');
        completedFor.current = '';
        adoptedFor.current = manifestName; // this session supersedes anything we might adopt
        setAdopted(false);
        setJobName(job.name);
        void resolveRun(job.name);
      },
      onError: toastApiError(toast, 'Failed to start benign-suite generation'),
    },
  });

  const { useJobsUpdateJobAttemptStatusDetails } = useHelixSdk();
  const { data: job } = usePolledJob(workspace, jobName);
  const details = job?.status_details as Record<string, unknown> | undefined;
  const interview = pendingInterview(details);
  const review = pendingReview(details);

  const patch = useJobsUpdateJobAttemptStatusDetails();
  const respond = (data: Record<string, unknown>) =>
    job && patch.mutate({ workspace, job: jobName, attempt: job.attempt_id, data });

  const status = job?.status;
  const terminal = Boolean(status && CJobTerminalStatuses.includes(status));
  const failure = useJobFailure('Benign-suite generation', jobName, job, { announce: !adopted });

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
    // Counted from the click, not from job creation, so a second click cannot queue a duplicate job.
    active: create.isPending || (Boolean(jobName) && !terminal),
    starting: create.isPending || (Boolean(jobName) && !runName),
    status,
    jobName,
    failure,
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
