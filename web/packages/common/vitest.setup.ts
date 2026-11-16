// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Common-specific vitest setup.
 * Shared browser mocks and matchers are provided by @nemo/testing/react/setup.
 *
 * https://vitest.dev/config/#setupfiles
 */
import failOnConsole from 'vitest-fail-on-console';

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
    if (message.includes('An update to CodeSnippetCode inside a test was not wrapped in act'))
      return true;
    return false;
  },
});
