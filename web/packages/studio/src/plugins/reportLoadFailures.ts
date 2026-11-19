// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { platformFetch } from '@nemo/sdk/src/utils/platformRequest';
import type { PluginLoadFailure, PluginLoadIssue } from '@studio/plugins/types';

export const PLUGIN_LOAD_ERRORS_ENDPOINT = '/studio/api/plugin-load-errors';

// Limits of POST /studio/api/plugin-load-errors, the same numbers as
// services/studio/src/nhx/studio/plugin_reports.py: keep in sync. Over-limit
// bodies are a silent 422.
export const REPORT_LIMITS = {
  issues: 10,
  subject: 200,
  message: 500,
  steps: 8,
  step: 300,
} as const;

const cut = (text: string, max: number) => text.slice(0, max);

export const clampIssues = (issues: PluginLoadIssue[]): PluginLoadIssue[] =>
  issues.slice(0, REPORT_LIMITS.issues).map((i) => ({
    code: i.code,
    subject: cut(i.subject, REPORT_LIMITS.subject),
    message: cut(i.message, REPORT_LIMITS.message),
    steps: i.steps.slice(0, REPORT_LIMITS.steps).map((s) => cut(s, REPORT_LIMITS.step)),
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
