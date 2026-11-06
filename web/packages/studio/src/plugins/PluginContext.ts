// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { LoadedPlugin, PluginContextValue, PluginLoadFailure } from '@studio/plugins/types';
import { createContext, useContext } from 'react';

export const PluginContext = createContext<PluginContextValue>({
  plugins: [],
  failed: [],
  installedNames: new Set(),
  isLoaded: false,
  isError: false,
});

export const usePlugins = (): LoadedPlugin[] => useContext(PluginContext).plugins;
export const usePluginsLoaded = (): boolean => useContext(PluginContext).isLoaded;
/** Returns true if the plugin manifest could not be fetched. */
export const usePluginsError = (): boolean => useContext(PluginContext).isError;
/** Returns true if the named plugin is registered in /apis/plugins (with or without a bundle). */
export const usePluginInstalled = (name: string): boolean =>
  useContext(PluginContext).installedNames.has(name);
/** Plugins with a web bundle that failed to load, with the reasons. */
export const usePluginLoadFailures = (): PluginLoadFailure[] => useContext(PluginContext).failed;
/** The load failure for the named plugin, if it has one. */
export const usePluginLoadFailure = (name: string): PluginLoadFailure | undefined =>
  useContext(PluginContext).failed.find((f) => f.name === name);
