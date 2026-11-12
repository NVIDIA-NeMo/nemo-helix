// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Me } from '@nemo/sdk/generated/platform/schema';
import { EditProfileModal } from '@studio/components/UserPopover/EditProfileModal';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const { mockMutateAsync } = vi.hoisted(() => ({ mockMutateAsync: vi.fn() }));

vi.mock('@nemo/sdk/generated/platform/profiles', () => ({
  getAuthGetMeQueryKey: () => ['me'],
  useAuthUpdateMyProfile: () => ({ mutateAsync: mockMutateAsync, isPending: false, error: null }),
}));

describe('EditProfileModal', () => {
  it('should refresh cached principal display names after saving', async () => {
    const me = { email: 'a@x', profile: { display_name_override: null } } as unknown as Me;
    mockMutateAsync.mockResolvedValue(me);
    const queryClient = new QueryClient();
    const stale = ['auth', 'principal-display', 'a@x'];
    queryClient.setQueryData(stale, { display_name: 'Old' });
    render(
      <QueryClientProvider client={queryClient}>
        <EditProfileModal me={me} onClose={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.type(screen.getByRole('textbox'), 'Alice A.');
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));

    await waitFor(() => expect(queryClient.getQueryState(stale)?.isInvalidated).toBe(true));
  });
});
