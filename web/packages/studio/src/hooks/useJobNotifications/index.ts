// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { jsonFilter } from '@nemo/common/src/api/filterOperators';
import { JOB_POLLING_INTERVAL_LONG } from '@nemo/common/src/constants';
import { HelixJobTerminalStatuses } from '@nemo/common/src/constants/query';
import { useJobsListJobs } from '@nemo/sdk/generated/platform/jobs';
import {
  HelixJobListSortField,
  type HelixJobsListFilter,
} from '@nemo/sdk/generated/platform/schema';
import { getHiddenJobSources } from '@studio/components/dataViews/JobsDataView/constants';
import {
  isNotificationGranted,
  useNotificationPermission,
} from '@studio/hooks/useNotificationPermission';
import {
  buildJobNotification,
  findJobTransitions,
  getJobNotificationPath,
  type NotifiableJob,
} from '@studio/util/jobNotifications';
import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router';

const PAGE_SIZE = 50;
/** Look-back from mount, so a job that finishes between two polls is still reported. */
const LOOKBACK_MS = 60_000;

/** Only worth interrupting someone who is not already looking at Studio. */
const isUserAway = (): boolean => document.visibilityState === 'hidden' || !document.hasFocus();

/**
 * Raises a desktop notification when a job that was running finishes or fails.
 *
 * Polls the workspace's jobs (unfinished ones, plus any updated since the page opened) and
 * diffs them against the jobs already seen running. Only jobs watched from an unfinished
 * state notify, and only while the tab is hidden or unfocused. Clicking a notification focuses
 * the window and opens the job's page.
 */
export const useJobNotifications = (workspace: string | undefined): void => {
  const navigate = useNavigate();
  const { requestIfNeeded } = useNotificationPermission();
  const [since] = useState(() => new Date(Date.now() - LOOKBACK_MS).toISOString());
  const watched = useRef<{ workspace: string; names: ReadonlySet<string> }>({
    workspace: '',
    names: new Set(),
  });

  const { data, dataUpdatedAt } = useJobsListJobs(
    workspace ?? '',
    {
      page: 1,
      page_size: PAGE_SIZE,
      sort: HelixJobListSortField['-created_at'],
      filter: jsonFilter<HelixJobsListFilter>({
        $and: [
          { source: { $nin: getHiddenJobSources() } },
          {
            $or: [{ status: { $nin: HelixJobTerminalStatuses } }, { updated_at: { $gte: since } }],
          },
        ],
      }),
    },
    {
      query: {
        enabled: Boolean(workspace),
        staleTime: 0,
        // The whole point is to notice a job finishing while the user is on another tab, and
        // TanStack Query pauses interval refetching in a background tab unless told not to.
        refetchIntervalInBackground: true,
        refetchInterval: (query) =>
          query.state.status === 'error' ? false : JOB_POLLING_INTERVAL_LONG,
      },
    }
  );

  useEffect(() => {
    if (!workspace || !data) return;
    // Another workspace's jobs are a fresh slate, not "finished" ones.
    const previous =
      watched.current.workspace === workspace ? watched.current.names : new Set<string>();
    const jobs: NotifiableJob[] = data.data;
    const { finished, watched: names } = findJobTransitions(jobs, previous);
    watched.current = { workspace, names };

    // First running job seen: ask now, while the user has just started something long.
    if (previous.size === 0 && names.size > 0) void requestIfNeeded();

    if (!isNotificationGranted() || !isUserAway()) return;
    for (const job of finished) {
      const content = buildJobNotification(job);
      if (!content) continue;
      try {
        const notification = new Notification(content.title, {
          body: content.body,
          tag: content.tag,
          // Ask the OS to play its notification sound. Browsers have no custom-sound option;
          // whether a sound is actually heard still follows the OS's notification settings.
          silent: false,
        });
        notification.onclick = () => {
          window.focus();
          void navigate(getJobNotificationPath(job, workspace));
          notification.close();
        };
      } catch {
        // Constructing a notification can throw on platforms that only allow service workers.
      }
    }
    // `dataUpdatedAt` is the trigger: an identical poll result is still a new poll.
  }, [dataUpdatedAt, data, navigate, requestIfNeeded, workspace]);
};
