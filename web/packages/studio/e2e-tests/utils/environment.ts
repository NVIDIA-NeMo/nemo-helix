// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { loadEnv } from 'vite';

const mode = process.env.MODE ?? 'e2e';

export const {
  VITE_E2E_PROJECT_NAME: E2E_PROJECT_NAME,
  VITE_NHX_BASE_URL: NHX_BASE_URL,
  VITE_FF_INTAKE_ENABLED,
  VITE_FF_TRACE_GRAPH_ENABLED,
  VITE_PLATFORM_BASE_URL: PLATFORM_BASE_URL,
} = loadEnv(mode, 'env');

/**
 * Studio URL for E2E tests.
 * - In CI with ephemeral deployments: Uses VSERVICE_URL_STUDIO_UI from the deployment job
 * - Local development: Falls back to local development server
 */
export const STUDIO_URL = process.env.VSERVICE_URL_STUDIO_UI || 'https://localhost:5173';

/**
 * User ID. This can be set when running tests locally; if populated, the namespace for test resources
 * created by tests will be prepended with this string.
 */
export const USER_ID = process.env.USER_ID;

/**
 * True if Studio routes and elements that rely on Intake are enabled.
 */
export const INTAKE_ENABLED = VITE_FF_INTAKE_ENABLED !== 'false';

/**
 * True if the Intake trace graph view is enabled.
 */
export const TRACE_GRAPH_ENABLED = VITE_FF_TRACE_GRAPH_ENABLED === 'true';

/**
 * The base path of STUDIO_URL,
 * i.e. if STUDIO_URL is http://localhost:8080/studio/,
 * STUDIO_URL_BASE_PATH will be /studio
 * */
export const STUDIO_URL_BASE_PATH = new URL(STUDIO_URL).pathname;
