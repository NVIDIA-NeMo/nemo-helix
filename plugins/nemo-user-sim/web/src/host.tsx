// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createContext, useContext, type ReactNode } from 'react';
import type { PluginHost } from './types';

const HostContext = createContext<PluginHost | null>(null);

export const HostProvider = ({ host, children }: { host: PluginHost; children: ReactNode }) => (
  <HostContext.Provider value={host}>{children}</HostContext.Provider>
);

export const useHost = (): PluginHost => {
  const host = useContext(HostContext);
  if (!host) throw new Error('useHost must be used within the user-sim plugin Root');
  return host;
};
