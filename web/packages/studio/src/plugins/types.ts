// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ComponentType, ReactNode } from 'react';

/**
 * Studio's SDK, by service. Plugins call these hooks directly; they dispatch into
 * Studio's React (a shared singleton), running on Studio's axios + QueryClient.
 * Passed by prop so plugins need no build dependency on the private `@nemo/sdk`.
 */
export interface PluginSdk {
  platform: typeof import('@nemo/sdk/generated/platform/index');
  agents: typeof import('@nemo/sdk/generated/agents/index');
}

/** Navigate Studio's shared router; paths are absolute Studio routes. */
export interface PluginNavigation {
  navigate: (to: string) => void;
  back: () => void;
}

export type NotificationType = 'success' | 'error' | 'info' | 'warning';

export interface NotificationOptions {
  /** ms before auto-dismiss; `false` keeps the toast until dismissed. Defaults per type. */
  durationMs?: number | false;
}

/** Fire a toast into Studio's shared toaster; defaults to `info`. */
export interface PluginNotifications {
  notify: (message: string, type?: NotificationType, options?: NotificationOptions) => void;
}

export interface PluginBreadcrumb {
  label: string;
  /** Absolute Studio path; omit for the trailing (current) crumb. */
  href?: string;
}

/**
 * Write into Studio's breadcrumb bar, which renders in GlobalNav — outside the
 * plugin's own subtree, so a plugin cannot render it itself. Studio clears the
 * trail when the plugin unmounts.
 */
export interface PluginBreadcrumbs {
  set: (trail: PluginBreadcrumb[]) => void;
}

/** Structured logging to Studio's OTEL pipeline, auto-scoped to the plugin. */
export interface PluginTelemetry {
  info: (message: string, cause?: unknown) => void;
  warn: (message: string, cause?: unknown) => void;
  error: (message: string, cause?: unknown) => void;
  event: (name: string, attributes?: Record<string, unknown>) => void;
}

/** The host handle Studio injects into every plugin; extend it to add capabilities. */
export interface PluginHost {
  workspaceId: string;
  /**
   * Origin the platform API is served from; empty when same-origin. A plugin
   * that calls its own service needs this: Studio's dev-server `/apis` proxy is
   * opt-in, so a relative request would otherwise hit the dev server whenever
   * VITE_PLATFORM_BASE_URL is set.
   */
  apiBaseUrl: string;
  // The configured platform bearer (access token or ID token); refresh tokens
  // must never cross the plugin boundary. Property names remain for compatibility.
  auth: {
    accessToken: string;
    getAccessToken: () => string;
  };
  sdk: PluginSdk;
  navigation: PluginNavigation;
  notifications: PluginNotifications;
  telemetry: PluginTelemetry;
  breadcrumbs: PluginBreadcrumbs;
}

export interface PluginRootProps {
  host: PluginHost;
}

/** API manifest returned by `GET /apis/plugins`. */
export interface PluginManifest {
  name: string;
  /** `null` for plugins registered without a web bundle. */
  bundleUrl: string | null;
}

/** A single navigation item contributed by a plugin. */
export interface PluginNavItem {
  id: string;
  /** Kebab-case Lucide icon name, e.g. `"flask-conical"`. */
  iconName: string;
  label: string;
  /**
   * Absolute path relative to the app root, e.g.
   * `/workspaces/:workspaceId/plugin/example/dashboard`.
   */
  href: string;
}

/** A group of navigation items contributed by a plugin. */
export interface PluginNavGroup {
  group: string;
  items: PluginNavItem[];
}

/**
 * A plugin bundle loaded at runtime via dynamic `import()`.
 *
 * The plugin's `Root` is rendered *inside* Studio's React tree — under the same
 * Router, QueryClient, and theme providers — so the plugin shares those contexts
 * (e.g. it navigates with Studio's router instead of standing up its own). The
 * plugin still ships as a separately-built bundle with its own private deps; only
 * the context-bearing singletons (react, react-dom, react-router) are shared via
 * the runtime import map.
 */
export interface LoadedPlugin {
  name: string;
  /** The plugin's root component, rendered within Studio's provider tree. */
  Root: ComponentType<PluginRootProps>;
  /** Return nav items scoped to the given workspace. */
  navItems: (workspaceId: string) => PluginNavGroup[];
}

/** The exports a loaded plugin bundle module must expose. */
export interface PluginModule {
  Root: LoadedPlugin['Root'];
  navItems: LoadedPlugin['navItems'];
}

/** Result of fetching the manifest and loading each plugin's bundle. */
export interface PluginQueryData {
  plugins: LoadedPlugin[];
  installedNames: ReadonlySet<string>;
}

/** Value exposed by the plugin React context. */
export interface PluginContextValue {
  plugins: LoadedPlugin[];
  /** All plugin names returned by /apis/plugins, including headless ones. */
  installedNames: ReadonlySet<string>;
  isLoaded: boolean;
  isError: boolean;
}

/** Props for the plugin context provider. */
export interface PluginProviderProps {
  children: ReactNode;
}
