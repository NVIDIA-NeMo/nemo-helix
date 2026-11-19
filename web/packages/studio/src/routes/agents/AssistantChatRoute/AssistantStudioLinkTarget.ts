// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BASE_URL } from '@studio/constants/environment';
import { DOCS_SITE_URL } from '@studio/constants/links';

const LEGACY_STUDIO_BASE_PATH = '/studio';
const DOCS_SITE = new URL(DOCS_SITE_URL);

/**
 * Returns the URL to open for a link into the NeMo docs site, or `undefined` for anything else.
 * Only https links on the docs site qualify, so URLs the model writes to other sites are never
 * made clickable.
 */
export const getAssistantDocsLinkTarget = (href: string | undefined): string | undefined => {
  if (!href) return undefined;

  let url: URL;
  try {
    url = new URL(href);
  } catch {
    return undefined;
  }

  const isDocsSite =
    url.protocol === 'https:' &&
    url.origin === DOCS_SITE.origin &&
    !url.username &&
    !url.password &&
    (url.pathname === DOCS_SITE.pathname || url.pathname.startsWith(`${DOCS_SITE.pathname}/`));
  return isDocsSite ? url.href : undefined;
};

const getNormalizedBaseUrl = (): string => {
  const normalized = BASE_URL.replace(/\/+$/, '');
  return normalized === '/' ? '' : normalized;
};

const isStudioRoute = (pathname: string): boolean =>
  pathname.startsWith('/workspaces/') || pathname === '/models' || pathname.startsWith('/models/');

const canonicalizeStudioPathname = (pathname: string): string => {
  const dashboardEvaluationPath = pathname.match(
    /^\/workspaces\/([^/]+)\/dashboard\/evaluations?(\/.*)?$/
  );
  if (dashboardEvaluationPath) {
    const [, workspace, remainder = ''] = dashboardEvaluationPath;
    return `/workspaces/${workspace}/evaluation${remainder}`;
  }

  const pluralEvaluationPath = pathname.match(/^\/workspaces\/([^/]+)\/evaluations(\/.*)?$/);
  if (pluralEvaluationPath) {
    const [, workspace, remainder = ''] = pluralEvaluationPath;
    return `/workspaces/${workspace}/evaluation${remainder}`;
  }

  return pathname;
};

const getStudioPathname = (pathname: string): string => {
  const baseUrl = getNormalizedBaseUrl();
  const pathWithoutBaseUrl =
    baseUrl && (pathname === baseUrl || pathname.startsWith(`${baseUrl}/`))
      ? pathname.slice(baseUrl.length) || '/'
      : pathname;

  if (pathWithoutBaseUrl === LEGACY_STUDIO_BASE_PATH) return '/';
  if (pathWithoutBaseUrl.startsWith(`${LEGACY_STUDIO_BASE_PATH}/`)) {
    return pathWithoutBaseUrl.slice(LEGACY_STUDIO_BASE_PATH.length);
  }

  return pathWithoutBaseUrl;
};

const rewriteWorkspacePath = (pathname: string, currentWorkspace: string | undefined): string => {
  if (!currentWorkspace || !pathname.startsWith('/workspaces/')) return pathname;

  const [, remainder] = pathname.match(/^\/workspaces\/[^/]+(\/.*)?$/) ?? [];
  return `/workspaces/${encodeURIComponent(currentWorkspace)}${remainder ?? ''}`;
};

const getWorkspaceDashboardPath = (currentWorkspace: string | undefined): string | undefined =>
  currentWorkspace ? `/workspaces/${encodeURIComponent(currentWorkspace)}/dashboard` : undefined;

export const getStudioInternalLinkTarget = (
  href: string | undefined,
  origin = window.location.origin,
  currentWorkspace?: string
): string | undefined => {
  if (!href) return undefined;

  let url: URL;
  try {
    url = new URL(href, origin);
  } catch {
    return undefined;
  }

  const pathname = canonicalizeStudioPathname(getStudioPathname(url.pathname));

  if (pathname === '/' || pathname === '') {
    const workspaceDashboardPath = getWorkspaceDashboardPath(currentWorkspace);
    return workspaceDashboardPath ? `${workspaceDashboardPath}${url.search}${url.hash}` : undefined;
  }

  if (!isStudioRoute(pathname)) return undefined;

  return `${rewriteWorkspacePath(pathname, currentWorkspace)}${url.search}${url.hash}`;
};
