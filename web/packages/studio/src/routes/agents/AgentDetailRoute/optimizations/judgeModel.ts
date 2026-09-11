// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getGatewayProxyGetQueryKey } from '@nemo/sdk/generated/platform/inference-gateway';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';

/** Bare entity name of a model ref, which may arrive qualified as `workspace/name`. The optimize
 *  job resolves the judge against the workspace it runs in, and its pre-flight looks the name up as
 *  a VirtualModel — a qualified name would miss. */
export const bareModelName = (modelRef: string): string =>
  modelRef.includes('/') ? modelRef.split('/').slice(1).join('/') : modelRef;

/**
 * OpenAI-compatible base URL the judge answers on.
 *
 * `/v1` rather than `/v1/chat/completions`: this lands in the Fabric model's `base_url`, and the
 * judge's client appends the route itself. Built from `PLATFORM_BASE_URL` the same way Studio
 * addresses agents when submitting an evaluation — the job resolves it server-side, so it has to be
 * a URL the platform answers on, not a browser-only one.
 */
export const judgeModelUrl = (workspace: string, modelRef: string): string => {
  const [path] = getGatewayProxyGetQueryKey(workspace, bareModelName(modelRef), 'v1');
  return `${PLATFORM_BASE_URL}${path}`;
};
