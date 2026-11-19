// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// These types are the plugin contract and must stay in sync with
// web/packages/studio/src/plugins/types.ts in the Studio monorepo.
// They are intentionally duplicated here so the plugin has no build-time
// dependency on Studio's internal packages.

// This plugin calls only its own service, so it needs nothing from Studio's SDK.
export type PluginSdk = unknown;

export interface Connection {
  name: string;
  provider: string;
  display_name: string;
  ui_prefix: string;
  frame_url: string;
}

export interface UISession {
  frame_url: string;
  expires_at: number;
}

export interface OTLPSnippet {
  yaml: string;
  upstream_setup: string[];
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
