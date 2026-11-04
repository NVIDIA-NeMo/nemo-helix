// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ToastObject } from '@nemo/common/src/providers/toast/types';

const DEFAULT_INTERVAL_MS = 2000;
const DEFAULT_TIMEOUT_MS = 120_000;

interface AnnounceCancellationOptions {
  toast: Pick<ToastObject, 'info' | 'success' | 'error'>;
  jobName: string;
  /** Status the cancel request itself returned. */
  status: string | undefined;
  /** Reads the job's current status. A rejection is treated as "try again". */
  fetchStatus: () => Promise<string | undefined>;
  intervalMs?: number;
  timeoutMs?: number;
}

/**
 * Tells the user what a cancel request did, in two steps.
 *
 * The cancel endpoint answers as soon as the request is accepted. A job with work in flight
 * is then `cancelling` until a controller stops it, so "cancelled" would overstate what the
 * response means. This says "requested" at once and announces the outcome when the job settles.
 * A job the request already cancelled gets the single success toast.
 *
 * Deliberately independent of any component: it holds only the toast and a fetch function,
 * so it keeps running after the row, menu, or page that started it is gone. It lives in
 * memory only; a page reload ends it, and the row's status badge is what remains.
 */
export const announceCancellation = async ({
  toast,
  jobName,
  status,
  fetchStatus,
  intervalMs = DEFAULT_INTERVAL_MS,
  timeoutMs = DEFAULT_TIMEOUT_MS,
}: AnnounceCancellationOptions): Promise<void> => {
  if (status === 'cancelled') {
    toast.success(`Job ${jobName} cancelled.`);
    return;
  }

  toast.info(`Cancellation requested for ${jobName}.`);

  // Only statuses the polls actually returned count: the request's own response is old news
  // by the deadline, and a failed poll tells us nothing.
  let lastSeen: string | undefined;
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    // Not Promise.withResolvers: Studio's tsconfig `lib` is ES2023, which lacks it.
    await new Promise<void>((resolve) => setTimeout(resolve, intervalMs));
    let current: string | undefined;
    try {
      current = await fetchStatus();
    } catch {
      continue;
    }
    lastSeen = current;
    if (current === 'cancelled') {
      toast.success(`Job ${jobName} cancelled.`);
      return;
    }
    if (current === 'error') {
      toast.error(`Job ${jobName} ended with an error while cancelling.`);
      return;
    }
    if (current === 'completed') {
      toast.info(`Job ${jobName} completed before it could be cancelled.`);
      return;
    }
  }

  toast.info(
    lastSeen === 'cancelling'
      ? `${jobName} is still cancelling. Check its status in the list.`
      : `Cancellation was requested for ${jobName}, but its status could not be confirmed. Check the list.`
  );
};
