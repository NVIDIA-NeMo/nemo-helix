// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  DEPLOYMENT_MODES_URL,
  deploymentModesResponse,
  EXECUTION_PROFILES_URL,
  executionProfilesResponse,
} from '@studio/mocks/handlers/agentDeploymentCapabilities';
import { server } from '@studio/mocks/node';
import { http, HttpResponse } from 'msw';

export const mockEnabledModes = (...enabled: string[]) =>
  server.use(
    http.get(DEPLOYMENT_MODES_URL, () => HttpResponse.json(deploymentModesResponse(enabled)))
  );

export const mockExecutionProfiles = (profiles: { profile: string; backend: string }[]) =>
  server.use(
    http.get(EXECUTION_PROFILES_URL, () => HttpResponse.json(executionProfilesResponse(profiles)))
  );
