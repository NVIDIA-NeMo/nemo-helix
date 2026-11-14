// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Studio-specific vitest setup.
 * Shared browser mocks and matchers are provided by @nemo/testing/react/setup.
 *
 * https://vitest.dev/config/#setupfiles
 *
 * Do not static-import @studio/mocks/node here: it pulls in the full MSW handlers graph. Load it once
 * in beforeAll via dynamic import so Vitest "setup" time (setupFiles evaluation) stays smaller.
 */
import type { SetupServer } from 'msw/node';
import failOnConsole from 'vitest-fail-on-console';
import 'blob-polyfill'; // Certain blob interactions from HF need this polyfill that aren't supported in JSDOM.

// Import mocks to prevent sign-in attempts and simplify editor testing
import '@studio/tests/mocks/react-oidc-context';
import '@studio/tests/mocks/Range';
import 'vitest-location-mock';

// Mock CodeSnippet to avoid Shiki creating multiple highlighter instances during tests.
// vi.mock() is hoisted above imports, so the factory must use dynamic import to load the
// shared mock implementation from @nemo/testing.
vi.mock('@nvidia/foundations-react-core', async (importOriginal) => {
  const { kuiFoundationsReactMock } = await import('@nemo/testing/mocks/kui');
  return kuiFoundationsReactMock(importOriginal);
});

vi.mock('react-router', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router')>();
  return {
    ...actual,
    useParams: vi.fn(actual.useParams),
    useNavigate: vi.fn(actual.useNavigate),
  };
});

/**
 * Fail tests on unexpected console.error/warn output.
 * Known, non-actionable messages are explicitly silenced below.
 * Any NEW console noise will cause a test failure, keeping output clean.
 */
failOnConsole({
  shouldFailOnError: true,
  shouldFailOnWarn: true,
  silenceMessage: (message) => {
    // React Router v6 blanket deprecation notices — no action possible until v7 upgrade.
    if (
      message.includes(
        'React Router Future Flag Warning: React Router will begin wrapping state updates in `React.startTransition` in v7.'
      )
    )
      return true;
    if (
      message.includes(
        'React Router Future Flag Warning: Relative route resolution within Splat routes is changing in v7.'
      )
    )
      return true;
    // KUI 1.0 emits a React.Fragment className warning from one of its primitives. Tracked in the
    // KUI upgrade ticket; silence so studio tests pass until the upstream fix lands.
    if (
      message.includes('Invalid prop `className` supplied to `React.Fragment`') ||
      message.includes('An update to CodeSnippetCode inside a test was not wrapped in act')
    )
      return true;
    return false;
  },
});

declare global {
  /** Set once per Vitest worker so MSW listen() is not repeated for every test file. */
  var __nemoStudioMswListening: boolean | undefined;
  /** MSW server instance (lazy-loaded). */
  var __nemoStudioMswServer: SetupServer | undefined;
}

// Start MSW once per worker (setupFiles run per test file; without a guard we listen/close in a loop)
beforeAll(async () => {
  if (globalThis.__nemoStudioMswListening) return;
  const { server } = await import('@studio/mocks/node');
  globalThis.__nemoStudioMswServer = server;
  server.listen({
    // This tells MSW to throw an error whenever it
    // encounters a request that doesn't have a
    // matching request handler.
    onUnhandledRequest: 'error',
  });
  globalThis.__nemoStudioMswListening = true;
});

afterEach(() => {
  globalThis.__nemoStudioMswServer?.resetHandlers();
});

// Intentionally no server.close(): Vitest setup runs per test file; closing after each file forced a
// full listen() on the next file and inflated "setup" time. The worker process exit tears down MSW.
