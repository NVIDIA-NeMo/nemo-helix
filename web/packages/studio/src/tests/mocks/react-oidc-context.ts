// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Export individual mock functions so tests can access them
export const mockSigninRedirect = vi.fn();
export const mockSignoutRedirect = vi.fn();
const mockRemoveUser = vi.fn();
const mockClearStaleState = vi.fn();
const mockSigninSilent = vi.fn();
const mockSigninPopup = vi.fn();
const mockSignoutSilent = vi.fn();
const mockSignoutPopup = vi.fn();
const mockQuerySessionStatus = vi.fn();
const mockRevokeTokens = vi.fn();

// Create the mock auth state at module level so it's accessible to vi.mock
const mockAuthState = {
  isAuthenticated: true,
  isLoading: false,
  user: {
    profile: {
      name: 'Test User',
      email: 'test@example.com',
      unique_name: 'test@example.com',
      sub: 'test-user-id',
      iss: 'test-issuer',
      aud: 'test-audience',
      exp: Math.floor(Date.now() / 1000) + 3600, // 1 hour from now
      iat: Math.floor(Date.now() / 1000),
    },
    access_token: 'mock-access-token',
    id_token: 'mock-id-token',
    refresh_token: 'mock-refresh-token',
    token_type: 'Bearer',
    expires_at: Math.floor(Date.now() / 1000) + 3600, // 1 hour from now (seconds, oidc-client-ts)
    scope: 'openid profile email',
  },
  activeNavigator: undefined,
  // Mock auth methods to prevent actual auth operations
  signinRedirect: mockSigninRedirect,
  signoutRedirect: mockSignoutRedirect,
  removeUser: mockRemoveUser,
  clearStaleState: mockClearStaleState,
  signinSilent: mockSigninSilent,
  signinPopup: mockSigninPopup,
  signoutSilent: mockSignoutSilent,
  signoutPopup: mockSignoutPopup,
  querySessionStatus: mockQuerySessionStatus,
  revokeTokens: mockRevokeTokens,
};

/**
 * Mock for react-oidc-context that provides safe auth state preventing sign-in attempts during tests.
 *
 * This mock ensures that:
 * - Tests never attempt to sign in (isAuthenticated: true)
 * - All auth methods are safely mocked
 * - Components receive realistic auth data
 */
vi.mock('react-oidc-context', () => ({
  useAuth: vi.fn(() => mockAuthState),
  AuthProvider: ({ children }: { children: React.ReactNode }) => children,
  hasAuthParams: vi.fn(() => false),
}));

export { mockAuthState };
