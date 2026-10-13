// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { setupDatasets, setupSafeSynthesizer, setupWorkspaces } from '@screenshots/setup';

export interface ViewportSize {
  width: number;
  height: number;
}

export interface ScreenshotConfig {
  /** Filename (without extension) — saved as {name}.png */
  name: string;
  /** Route to navigate to (relative to base URL) */
  route: string;
  /** Optional selector to wait for before capturing */
  waitFor?: string;
  /** Optional delay in ms after waitFor resolves (for animations) */
  delay?: number;
  /** Optional viewport clip region */
  clip?: { x: number; y: number; width: number; height: number };
  /** Optional per-screenshot viewport override */
  viewport?: ViewportSize;
  /** Optional setup function to interact with the page before capturing */
  setup?: SetupFn;
}

export type SetupFn = (page: import('@playwright/test').Page) => Promise<void>;

/**
 * Privacy replacements — text strings in the DOM that should be swapped
 * before each screenshot is captured. Useful for obscuring real user names,
 * emails, or other PII in documentation images.
 *
 * Keys are the original text, values are the replacement text.
 */
export const privacyReplacements: Record<string, string> = {
  'Octavian Drulea': 'Example User',
  odrulea: 'user',
  'Henrique Tolentino': 'Example User',
  htolentino: 'user',
};

/** Default viewport used for all screenshots unless overridden per-entry */
export const defaultViewport: ViewportSize = {
  width: 1200,
  height: 700,
};

/** Name of the demo workspace used across screenshots */
export const DEMO_WORKSPACE = 'demo';

/** Name of the demo dataset used across screenshots */
export const DEMO_DATASET = 'demo';

const ws = `/studio/workspaces/${DEMO_WORKSPACE}`;

export const screenshots: ScreenshotConfig[] = [
  {
    name: 'workspaces',
    route: `${ws}/dashboard`,
    setup: setupWorkspaces,
  },
  { name: 'dashboard', route: `${ws}/dashboard` },
  {
    name: 'datasets',
    route: `${ws}/datasets/${DEMO_WORKSPACE}%2F${DEMO_DATASET}`,
    setup: setupDatasets,
  },
  {
    name: 'safe-synthesizer',
    route: `${ws}/safe-synthesizer/new`,
    setup: setupSafeSynthesizer,
  },
  { name: 'jobs', route: `${ws}/jobs` },
  { name: 'secrets', route: `${ws}/secrets` },
  { name: 'fine-tune', route: `${ws}/fine-tune` },
  { name: 'prompt-tuned-models', route: `${ws}/customizations/prompt-tuned/new` },
  { name: 'run-evaluation', route: `${ws}/evaluation/results/new` },
  { name: 'evaluations', route: `${ws}/evaluation/results` },
];
