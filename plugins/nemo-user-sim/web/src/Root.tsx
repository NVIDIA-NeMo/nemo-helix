// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Route, Routes } from 'react-router';
import { HostProvider } from './host';
import { RunDetailRoute } from './RunDetailRoute';
import { RunListRoute } from './RunListRoute';
import type { PluginRootProps } from './types';

export const Root = ({ host }: PluginRootProps) => (
  <HostProvider host={host}>
    <Routes>
      <Route index element={<RunListRoute />} />
      <Route path=":job" element={<RunDetailRoute />} />
    </Routes>
  </HostProvider>
);
