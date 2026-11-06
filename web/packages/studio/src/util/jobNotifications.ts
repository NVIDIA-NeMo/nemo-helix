// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobTerminalStatuses } from '@nemo/common/src/constants/query';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { JOB_SOURCE } from '@studio/components/dataViews/JobsDataView/constants';
import { ROUTES } from '@studio/constants/routes';
import { getWorkspaceJobDetailRoute } from '@studio/routes/utils';
import { generatePath } from 'react-router';

/** The slice of a job the notification logic reads. */
export interface NotifiableJob {
  name: string;
  source: string;
  status: HelixJobStatus;
}

export interface JobNotificationContent {
  title: string;
  body: string;
  /** Same job, same tag: the browser replaces rather than stacks a repeat. */
  tag: string;
}

const isTerminal = (status: HelixJobStatus) => HelixJobTerminalStatuses.includes(status);

/**
 * Where a notification click should land. Sources with a dedicated detail page link to it;
 * everything else (evaluator, anonymizer, unknown) falls back to the generic job page, which
 * renders any job.
 */
export const getJobNotificationPath = (job: NotifiableJob, workspace: string): string => {
  switch (job.source) {
    case JOB_SOURCE.CUSTOMIZATION:
      return generatePath(ROUTES.workspace.customizationJobDetails, {
        workspace,
        customizationJobName: job.name,
      });
    case JOB_SOURCE.DATA_DESIGNER:
      return generatePath(ROUTES.workspace.dataDesignerJobDetails, {
        workspace,
        dataDesignerJobName: job.name,
      });
    case JOB_SOURCE.SAFE_SYNTHESIZER:
      return generatePath(ROUTES.workspace.safeSynthesizerJob, {
        workspace,
        safeSynthesizerJobName: job.name,
      });
    default:
      return getWorkspaceJobDetailRoute(workspace, job.name);
  }
};

/**
 * Content for a finished job, or `null` when no notification is due. A cancelled job is the
 * user's own doing, so it stays quiet.
 */
export const buildJobNotification = (job: NotifiableJob): JobNotificationContent | null => {
  if (job.status === HelixJobStatus.completed) {
    return { title: 'Job completed', body: `${job.name} finished successfully.`, tag: job.name };
  }
  if (job.status === HelixJobStatus.error) {
    return { title: 'Job failed', body: `${job.name} failed.`, tag: job.name };
  }
  return null;
};

export interface JobTransitions {
  /** Jobs that were being watched and have now finished, in list order. */
  finished: NotifiableJob[];
  /** Names to watch on the next poll: still-running jobs seen now, minus those that finished. */
  watched: Set<string>;
}

/**
 * Compares a poll against the jobs already being watched.
 *
 * A job is only ever notified about if it was first seen unfinished. That is what keeps a
 * freshly opened tab from announcing every job that had already finished.
 */
export const findJobTransitions = (
  jobs: readonly NotifiableJob[],
  previouslyWatched: ReadonlySet<string>
): JobTransitions => {
  const watched = new Set(previouslyWatched);
  const finished: NotifiableJob[] = [];
  for (const job of jobs) {
    if (!isTerminal(job.status)) {
      watched.add(job.name);
    } else if (watched.delete(job.name)) {
      finished.push(job);
    }
  }
  return { finished, watched };
};
