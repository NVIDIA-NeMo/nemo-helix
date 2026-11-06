// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { platformFetch } from '@nemo/sdk/src/utils/platformRequest';
import type { PluginLoadFailure, PluginLoadIssue } from '@studio/plugins/types';

export const PLUGIN_LOAD_ERRORS_ENDPOINT = '/studio/api/plugin-load-errors';

// Backend limits for POST /studio/api/plugin-load-errors; over-limit bodies are rejected (422).
const cut = (text: string, max: number) => text.slice(0, max);

export const clampIssues = (issues: PluginLoadIssue[]): PluginLoadIssue[] =>
  issues.slice(0, 10).map((i) => ({
    code: i.code,
    subject: cut(i.subject, 200),
    message: cut(i.message, 500),
    steps: i.steps.slice(0, 8).map((s) => cut(s, 300)),
  }));

/** Fire-and-forget: one report per failure; every error and non-2xx is ignored. */
export const reportLoadFailures = (baseUrl: string, failures: PluginLoadFailure[]): void => {
  for (const { name, issues } of failures) {
    platformFetch(`${baseUrl}${PLUGIN_LOAD_ERRORS_ENDPOINT}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ plugin: name, issues: clampIssues(issues) }),
    }).catch(() => {});
  }
};
