// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ResolvedProfile } from '@nemo/sdk/generated/platform/schema';
import { PrincipalName } from '@studio/components/PrincipalName';
import { render, screen } from '@testing-library/react';

const { mockDisplay, mockMe } = vi.hoisted(() => ({ mockDisplay: vi.fn(), mockMe: vi.fn() }));

vi.mock('@studio/hooks/usePrincipalDisplay', () => ({ usePrincipalDisplay: mockDisplay }));
vi.mock('@studio/providers/auth/useMe', () => ({ useMe: mockMe }));

const resolved = (profile: Partial<ResolvedProfile>) => ({
  data: {
    principal: 'alice@example.com',
    display_name_is_override: false,
    ambiguous: false,
    ...profile,
  },
});

describe('PrincipalName', () => {
  beforeEach(() => {
    mockMe.mockReturnValue({ data: undefined });
    mockDisplay.mockReturnValue({ data: undefined });
  });

  it('shows the raw principal until it resolves to a name', () => {
    render(<PrincipalName principal="alice@example.com" />);
    expect(screen.getByText('alice@example.com')).toBeInTheDocument();
  });

  it('always shows the principal beside a resolved name', () => {
    mockDisplay.mockReturnValue(resolved({ display_name: 'Alice Anderson' }));
    render(<PrincipalName principal="alice@example.com" layout="stacked" />);
    expect(screen.getByText('Alice Anderson')).toBeInTheDocument();
    expect(screen.getByText('alice@example.com')).toBeInTheDocument();
  });

  it('adds the identity provider name when the user overrode their name', () => {
    mockDisplay.mockReturnValue(
      resolved({
        display_name: 'Admin',
        idp_display_name: 'Alice Anderson',
        display_name_is_override: true,
      })
    );
    render(<PrincipalName principal="alice@example.com" />);
    expect(screen.getByText('Admin')).toHaveAttribute(
      'title',
      'alice@example.com · Alice Anderson'
    );
  });

  it('attributes creator-less resources to the local account when auth is off', () => {
    mockMe.mockReturnValue({
      data: { auth_method: 'disabled', profile: { display_name: 'Local user' } },
    });
    render(<PrincipalName principal={null} />);
    expect(screen.getByText('Local user')).toBeInTheDocument();
  });
});
