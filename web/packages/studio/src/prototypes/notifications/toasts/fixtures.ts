// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NAV_HREFS, type NavId } from '@studio/prototypes/notifications/fixtures';
import type {
  BriefToastOptions,
  NotifyOptions,
  ProtoNotificationsContextValue,
} from '@studio/prototypes/notifications/toasts/types';

export interface DemoEvent {
  label: string;
  sourceId: NavId;
  build: (api: ProtoNotificationsContextValue) => NotifyOptions;
}

const wait = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

export const JOB_COMPLETED: NotifyOptions = {
  status: 'success',
  heading: 'Fine-tuning job completed',
  description: 'llama-3.1-8b-instruct-lora finished training in 42m.',
  link: { label: 'View job', href: `${NAV_HREFS['custom-models']}/llama-3-1-8b-instruct-lora` },
};

export const EVALUATION_FAILED: NotifyOptions = {
  status: 'error',
  heading: 'Evaluation job failed',
  description: 'customer-support-eval stopped at step 312: CUDA out of memory.',
  link: {
    label: 'View logs',
    href: `${NAV_HREFS['evaluation-results']}/customer-support-eval?tab=logs`,
  },
};

export const GUARDRAIL_DEGRADED: NotifyOptions = {
  status: 'warning',
  heading: 'Guardrail latency degraded',
  description: 'content-safety p95 latency has been above 2s for 15 minutes.',
  link: { label: 'View guardrail', href: `${NAV_HREFS.guardrails}/content-safety` },
};

export const buildRoutingSuggestion = (
  api: Pick<ProtoNotificationsContextValue, 'notify' | 'toast'>
): NotifyOptions => ({
  status: 'info',
  heading: 'Routing suggestion for support-agent',
  description: 'Route 60% of traffic to nemotron-nano-9b. Projected cost −38%, accuracy −0.4 pts.',
  link: { label: 'Review', href: `${NAV_HREFS['agents-group']}/support-agent?tab=suggestions` },
  decision: {
    accept: {
      label: 'Apply',
      onAction: async () => {
        await wait(900);
        api.notify({
          status: 'success',
          heading: 'Routing applied',
          description: '60% of support-agent traffic now routes to nemotron-nano-9b.',
          link: { label: 'View agent', href: `${NAV_HREFS['agents-group']}/support-agent` },
        });
      },
    },
    decline: {
      label: 'Dismiss',
      onAction: async () => {
        await wait(400);
        api.toast({ status: 'info', message: 'Suggestion dismissed' });
      },
    },
  },
});

export const DATASET_UPLOADED: NotifyOptions = {
  status: 'success',
  heading: 'Dataset uploaded',
  description: 'support-tickets is ready to use.',
  link: { label: 'View dataset', href: `${NAV_HREFS['dataset-list']}/support-tickets` },
};

export const SETTINGS_SAVED: BriefToastOptions = {
  status: 'success',
  message: 'Settings saved',
};

export const DEMO_EVENTS: DemoEvent[] = [
  { label: 'Fine-tuning job completed', sourceId: 'custom-models', build: () => JOB_COMPLETED },
  {
    label: 'Evaluation job failed',
    sourceId: 'evaluation-results',
    build: () => EVALUATION_FAILED,
  },
  { label: 'Agent routing suggestion', sourceId: 'agents-group', build: buildRoutingSuggestion },
  { label: 'Guardrail latency degraded', sourceId: 'guardrails', build: () => GUARDRAIL_DEGRADED },
];
