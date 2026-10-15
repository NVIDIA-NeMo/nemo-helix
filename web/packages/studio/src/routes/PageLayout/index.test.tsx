// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PageLayout } from '@studio/routes/PageLayout';
import { render, screen } from '@testing-library/react';

const mocks = vi.hoisted(() => ({
  useAuthAutoLogin: vi.fn(),
}));

vi.mock('@studio/providers/auth', () => ({
  useAuthAutoLogin: mocks.useAuthAutoLogin,
}));

vi.mock('@studio/providers/auth/useAuthTokenStatus', () => ({
  useAuthTokenStatus: () => ({ isTokenActive: false }),
}));

vi.mock('@studio/providers/workspace', () => ({
  useSelectedWorkspace: () => ({
    selectedWorkspace: undefined,
    isWorkspaceUnauthorized: false,
    isWorkspaceLoading: false,
  }),
}));

vi.mock('@studio/hooks/useWorkspaceFromPath', () => ({
  useWorkspaceFromPathIfExists: () => undefined,
}));

describe('PageLayout authentication state', () => {
  it('shows progress while authentication discovery is pending', () => {
    mocks.useAuthAutoLogin.mockReturnValue({
      isAuthError: false,
      isAuthPending: true,
      isSignedOut: false,
      signIn: vi.fn(),
    });

    render(<PageLayout />);

    expect(screen.getByRole('main')).toHaveAttribute('aria-busy', 'true');
    expect(screen.getByText('Checking sign-in status...')).toBeInTheDocument();
  });
});
