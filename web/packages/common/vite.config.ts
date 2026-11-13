// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { baseTestConfig } from '@nemo/testing/react/config';
import react from '@vitejs/plugin-react';
import svgr from 'vite-plugin-svgr';
import { defineConfig, mergeConfig } from 'vitest/config';

// eslint-disable-next-line import/no-default-export
export default mergeConfig(
  baseTestConfig,
  defineConfig({
    plugins: [react(), svgr()],
    resolve: {
      tsconfigPaths: true,
    },
    test: {
      setupFiles: ['@nemo/testing/react/setup', './vitest.setup.ts'],
      globalSetup: '@nemo/testing/react/global-setup',
      coverage: {
        include: ['src/**/*.{js,jsx,ts,tsx}'],
        // ajv-generated validator: not hand-written, not meaningfully coverable
        exclude: ['src/utils/jsonSchema/generated/**'],
        provider: 'v8',
        reporter: ['text', 'html'],
        reportsDirectory: '.test-reports/coverage',
      },
    },
  })
);
