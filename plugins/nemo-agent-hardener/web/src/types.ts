// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// These types are the plugin contract and must stay in sync with
// web/packages/studio/src/plugins/types.ts in the Studio monorepo.
// They are intentionally duplicated here so the plugin has no build-time
// dependency on Studio's internal packages.

/**
 * Typed exactly as Studio types it. This is a *type-only* import — erased at
 * compile time, so the private `@nemo/sdk` package is never bundled or
 * installed; only `paths` in tsconfig.json resolves it. Studio passes the real
 * module on `host.sdk`, running on its own axios + QueryClient.
 */
export interface PluginSdk {
  platform: typeof import('@nemo/sdk/generated/platform/api');
  agents: typeof import('@nemo/sdk/generated/agents/api');
}

export interface PluginNavigation {
  navigate: (to: string) => void;
  back: () => void;
}

export type NotificationType = 'success' | 'error' | 'info' | 'warning';

export interface NotificationOptions {
  durationMs?: number | false;
}

export interface PluginNotifications {
  notify: (message: string, type?: NotificationType, options?: NotificationOptions) => void;
}

export interface PluginBreadcrumb {
  label: string;
  href?: string;
}

export interface PluginBreadcrumbs {
  set: (trail: PluginBreadcrumb[]) => void;
}

export interface PluginTelemetry {
  info: (message: string, cause?: unknown) => void;
  warn: (message: string, cause?: unknown) => void;
  error: (message: string, cause?: unknown) => void;
  event: (name: string, attributes?: Record<string, unknown>) => void;
}

export interface PluginHost {
  workspaceId: string;
  /**
   * Origin the platform API is served from, empty when same-origin. Studio's
   * dev proxy is opt-in, so a plugin cannot assume relative `/apis/...` reaches
   * the platform — prefix requests with this.
   */
  apiBaseUrl: string;
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

export interface PluginNavItem {
  id: string;
  iconName: string;
  label: string;
  href: string;
}

export interface PluginNavGroup {
  group: string;
  items: PluginNavItem[];
}
