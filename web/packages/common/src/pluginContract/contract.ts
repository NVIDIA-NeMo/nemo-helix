// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PluginContract } from '@nemo/common/src/pluginContract/types';

// singleton: the plugin must resolve to Studio's instance (identity matters).
// provided: only API compatibility matters. Studio serves a separate vendor build of
// Common's source, so state crosses via the `host` prop, not module identity.
export const PLUGIN_CONTRACT: PluginContract = {
  version: '1.0.0',
  entries: [
    { specifier: 'react', package: 'react', kind: 'singleton' },
    { specifier: 'react/jsx-runtime', package: 'react', kind: 'singleton' },
    { specifier: 'react-dom', package: 'react-dom', kind: 'singleton' },
    { specifier: 'react-dom/client', package: 'react-dom', kind: 'singleton' },
    { specifier: 'react-router', package: 'react-router', kind: 'singleton' },
    {
      specifier: '@nvidia/foundations-react-core',
      package: '@nvidia/foundations-react-core',
      kind: 'singleton',
    },
    { specifier: '@tanstack/react-query', package: '@tanstack/react-query', kind: 'singleton' },
    { specifier: '@nemo/common', package: '@nemo/common', kind: 'provided' },
  ],
};
