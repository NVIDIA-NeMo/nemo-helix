// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { configureLogger } from '@nemo/common/src/utils/logger';
import {
  ThemeProvider as KaizenThemeProvider,
  Theme,
  TooltipProvider,
} from '@nvidia/foundations-react-core';
import { queryClient } from '@studio/api/queryClient';
import {
  AUTH_AUTHORITY,
  AUTH_CLIENT_ID,
  AUTH_SCOPE_PREFIX,
  AUTH_SCOPES,
  BASE_URL,
  OTEL_SERVICE_NAME,
} from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { routes } from '@studio/routes';
import { useLocalStorage } from '@studio/util/hooks/useLocalStorage';
import { UI_THEME } from '@studio/util/localStorage';
import { logVersion } from '@studio/util/logVersion';
import { QueryClientProvider } from '@tanstack/react-query';
import { WebStorageStateStore } from 'oidc-client-ts';
import { StrictMode } from 'react';
import { AuthProvider } from 'react-oidc-context';
import { createBrowserRouter } from 'react-router';
import { RouterProvider } from 'react-router/dom';

/**
 * Expand OAuth scopes by prepending scope_prefix to custom scopes.
 *
 * Standard OIDC scopes (openid, profile, email, offline_access) are kept as-is.
 * Custom scopes containing ':' or ending with '.default' get the prefix prepended.
 * This matches the logic in the CLI (sdk/.../cli/commands/auth.py).
 */
const expandScopes = (scopes: string, scopePrefix?: string): string => {
  if (!scopePrefix) return scopes;
  const prefix = scopePrefix.endsWith('/') ? scopePrefix : `${scopePrefix}/`;
  return scopes
    .split(/\s+/)
    .filter(Boolean)
    .map((s) => (s.includes(':') || s.endsWith('.default') ? `${prefix}${s}` : s))
    .join(' ');
};

// Note: there is a bug in react-router that does not clean up blockers correctly,
// so that if a new router is created while a blocker is active, you will see a
// warning about 'A router only supports one blocker at a time'. https://github.com/remix-run/react-router/issues/11430
// This will only affect local development for us because we only create the router
// once and do not dynamically set it up / tear it down like some other people
// who are experiencing the problem, and react-router's own blocking example
// manually disposes of the router upon hot reload, so we are folowing their lead.
// https://github.com/remix-run/react-router/blob/main/examples/navigation-blocking/src/app.tsx#L42C1-L44C2
if (import.meta.hot) {
  import.meta.hot.dispose(() => router.dispose());
}

const router = createBrowserRouter(routes, { basename: BASE_URL ?? '/' });
configureLogger(OTEL_SERVICE_NAME);
logVersion();

const effectiveScope = expandScopes(AUTH_SCOPES || 'openid profile email', AUTH_SCOPE_PREFIX);

const redirectUri = [
  window.location.origin,
  ...BASE_URL.split('/'),
  ...ROUTES.auth.success.split('/'),
]
  .filter(Boolean)
  .join('/');

const userStore = new WebStorageStateStore({ store: window.localStorage });

export const App = () => {
  const [savedTheme] = useLocalStorage<Theme>(UI_THEME, 'dark');

  return (
    <StrictMode>
      <KaizenThemeProvider
        global
        density="standard"
        theme={savedTheme}
        // eslint-disable-next-line no-restricted-syntax
        style={{ height: '100%' }}
      >
        <QueryClientProvider client={queryClient}>
          <AuthProvider
            client_id={AUTH_CLIENT_ID}
            redirect_uri={redirectUri}
            authority={AUTH_AUTHORITY}
            scope={effectiveScope}
            userStore={userStore}
            automaticSilentRenew
          >
            <TooltipProvider>
              <RouterProvider router={router} />
            </TooltipProvider>
          </AuthProvider>
        </QueryClientProvider>
      </KaizenThemeProvider>
    </StrictMode>
  );
};
