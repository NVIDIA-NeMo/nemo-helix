/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

export const isUnroutableHost = (url: string): boolean => {
  try {
    const host = new URL(url).hostname
      .replace(/^\[|\]$/g, '')
      .replace(/\.$/, '')
      .toLowerCase();
    const isWildcardHost = host === '0.0.0.0' || host === '::' || host === '0:0:0:0:0:0:0:0';
    // Browsers outside Kubernetes cannot resolve Service DNS names. Dotless
    // non-localhost names are treated as likely short Service names.
    const isDotlessNonLocalhost =
      host !== 'localhost' && !host.includes('.') && !host.includes(':');
    const isLikelyClusterInternalHost =
      host.endsWith('.svc') || host.includes('.svc.') || isDotlessNonLocalhost;
    return isWildcardHost || isLikelyClusterInternalHost;
  } catch {
    return false;
  }
};

export const resolveBrowserBaseUrl = (envValue: string | undefined): string => {
  if (envValue && !isUnroutableHost(envValue)) return envValue;
  return typeof window !== 'undefined' ? window.location.origin : '';
};
