// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  resolve: {
    // Array form: ordered, so the deep prefix is matched before the bare name.
    alias: [
      {
        find: '@agent-hardener',
        replacement: fileURLToPath(new URL('./src', import.meta.url)),
      },
      // `@nemo/common` is externalized in the build (Studio serves it through
      // the import map), so tests resolve it from source the way tsconfig does.
      // The barrel re-exports via its own deep `@nemo/common/src/...` paths,
      // which need the prefix alias below.
      {
        find: /^@nemo\/common\/(.*)$/,
        replacement: fileURLToPath(new URL('../../../web/packages/common/$1', import.meta.url)),
      },
      {
        find: /^@nemo\/common$/,
        replacement: fileURLToPath(
          new URL('../../../web/packages/common/src/plugin.ts', import.meta.url)
        ),
      },
    ],
  },
  test: {
    globals: true,
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
});
