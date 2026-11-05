// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { http, HttpResponse } from 'msw';

export const AGENT_IMAGES_URL = `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agent-images`;

export const agentImagesHandlers = [
  http.get(AGENT_IMAGES_URL, () => HttpResponse.json({ data: [] })),
  http.delete(`${AGENT_IMAGES_URL}/:imageId`, () => new HttpResponse(null, { status: 204 })),
];
