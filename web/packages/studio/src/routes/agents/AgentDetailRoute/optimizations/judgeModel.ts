// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getGatewayProxyGetQueryKey } from '@nemo/sdk/generated/platform/inference-gateway';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';

export const bareModelName = (modelRef: string): string =>
  modelRef.includes('/') ? modelRef.split('/').slice(1).join('/') : modelRef;

/**
 * OpenAI-compatible base URL the judge answers on.
 */
export const judgeModelUrl = (workspace: string, modelRef: string): string => {
  const [path] = getGatewayProxyGetQueryKey(workspace, bareModelName(modelRef), 'v1');
  return `${PLATFORM_BASE_URL}${path}`;
};
