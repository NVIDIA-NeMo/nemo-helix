// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import path from 'path';
import { fileURLToPath } from 'url';
import type { StorybookConfig } from '@storybook/react-vite';
import svgr from 'vite-plugin-svgr';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const config: StorybookConfig = {
  stories: ['../../studio/src/**/*.stories.@(ts|tsx)', '../../common/src/**/*.stories.@(ts|tsx)'],
  staticDirs: ['../public'],
  addons: ['@storybook/addon-a11y'],
  framework: {
    name: '@storybook/react-vite',
    options: {},
  },
  viteFinal: async (viteConfig) => {
    return {
      ...viteConfig,
      plugins: [...(viteConfig.plugins ?? []), svgr()],
      resolve: {
        ...viteConfig.resolve,
        alias: {
          ...viteConfig.resolve?.alias,
          '@studio': path.resolve(__dirname, '../../studio/src'),
          '@nemo/common': path.resolve(__dirname, '../../common'),
        },
      },
      css: {
        ...viteConfig.css,
        postcss: {
          plugins: [
            // Tailwind for component styles; base points at workspace packages for @source
            (await import('@tailwindcss/postcss')).default({
              base: path.resolve(__dirname, '../..'), // packages directory
            }),
          ],
        },
      },
    };
  },
};

export default config;
