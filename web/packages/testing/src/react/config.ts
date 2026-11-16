// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Shared base vitest configuration for React/DOM packages.
 *
 * Usage in a consuming package's vite.config.ts:
 *   import { baseTestConfig } from '@nemo/testing/react/config';
 *   import { mergeConfig, defineConfig } from 'vitest/config';
 *   export default mergeConfig(baseTestConfig, defineConfig({ ... }));
 */
import { defineConfig } from 'vitest/config';

export const baseTestConfig = defineConfig({
  test: {
    environment: 'happy-dom',
    globals: true,
  },
});
