// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Only Platform-managed agents can be packaged; NAT workflows build from a checkout. */
export const FABRIC_CONFIG_FORMAT = 'nemo-agents-spec-v1';

/** Matches `PACKAGE_RESULT_NAME` in the nemo-agents plugin. */
export const PACKAGE_RESULT_NAME = 'package_result';

/**
 * The jobs scheduler dispatches within seconds, so staying `created` past this
 * means nothing is picking the job up — usually a platform started without the
 * jobs controller, where the UI would otherwise imply progress forever.
 */
export const QUEUED_STALL_MS = 60_000;

const TERMINAL_JOB_STATUSES = new Set(['completed', 'error', 'cancelled']);

export const isTerminalPackageStatus = (status: string | undefined): boolean =>
  status !== undefined && TERMINAL_JOB_STATUSES.has(status);

export const isQueuedTooLong = (
  status: string | undefined,
  startedAt: number | undefined,
  now: number
): boolean => status === 'created' && startedAt !== undefined && now - startedAt > QUEUED_STALL_MS;

export interface PackageResult {
  image: string;
  agent: string;
  published: string;
}

/**
 * Narrow the `package_result` artifact, which is untrusted JSON off the wire. A
 * result without a usable `image` is absent rather than a blank tag the user
 * could paste into a deployment.
 */
export const parsePackageResult = (parsed: unknown): PackageResult | undefined => {
  const image = (parsed as { image?: unknown })?.image;
  if (typeof image !== 'string' || !image.trim()) return undefined;
  const agent = (parsed as { agent?: unknown })?.agent;
  const published = (parsed as { published?: unknown })?.published;
  return {
    image: image.trim(),
    agent: typeof agent === 'string' ? agent : '',
    published: typeof published === 'string' ? published : '',
  };
};

export const parseJobTimestamp = (value: string | undefined): number | undefined => {
  if (!value) return undefined;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? undefined : parsed;
};
