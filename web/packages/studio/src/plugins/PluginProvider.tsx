// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  NO_FAILURES,
  NO_NAMES,
  NO_PLUGINS,
  PLUGINS_MANIFEST_QUERY_KEY,
} from '@studio/plugins/consts';
import { PluginContext } from '@studio/plugins/PluginContext';
import type { PluginProviderProps } from '@studio/plugins/types';
import { fetchPlugins } from '@studio/plugins/utils';
import { useQuery } from '@tanstack/react-query';

export const PluginProvider = ({ children }: PluginProviderProps) => {
  const { data, isSuccess, isError } = useQuery({
    queryKey: PLUGINS_MANIFEST_QUERY_KEY,
    queryFn: fetchPlugins,
    staleTime: Infinity,
    gcTime: Infinity,
    refetchOnReconnect: false,
  });

  return (
    <PluginContext.Provider
      value={{
        plugins: data?.plugins ?? NO_PLUGINS,
        failed: data?.failed ?? NO_FAILURES,
        installedNames: data?.installedNames ?? NO_NAMES,
        isLoaded: isSuccess || isError,
        isError,
      }}
    >
      {children}
    </PluginContext.Provider>
  );
};
