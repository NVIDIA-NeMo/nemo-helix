// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// These types are the plugin contract and must stay in sync with
// web/packages/studio/src/plugins/types.ts in the Studio monorepo.
// They are intentionally duplicated here so the example plugin has no
// build-time dependency on Studio's internal packages.

// Minimal mirror of the workspace fields this example renders.
export interface Workspace {
  name: string;
  status?: string;
  created_at?: string;
}

// Minimal mirror of Studio's PluginSdk — only the hooks this example calls, so it
// stays free of the private @nemo/sdk package.
export interface PluginSdk {
  platform: {
    useEntitiesListWorkspaces: (
      params?: { page?: number; page_size?: number },
      options?: { query?: { enabled?: boolean; staleTime?: number } }
    ) => {
      data?: { data?: Workspace[] };
      isPending: boolean;
      isError: boolean;
    };
  };
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
  /** Origin the platform API is served from; empty when same-origin. */
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
