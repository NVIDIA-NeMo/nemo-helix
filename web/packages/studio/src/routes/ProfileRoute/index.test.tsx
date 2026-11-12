// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Me, ProfileUpdate } from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { mockMe } from '@studio/mocks/handlers/profiles';
import { server } from '@studio/mocks/node';
import { ProfileRoute } from '@studio/routes/ProfileRoute';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { createMemoryRouter, RouterProvider } from 'react-router';

vi.mock('@studio/providers/auth/useAuthTokenStatus', () => ({
  useAuthTokenStatus: () => ({ isTokenActive: true }),
}));

const ME_URL = `${PLATFORM_BASE_URL}/apis/auth/v2/me`;

const renderRoute = (me: Me = mockMe) => {
  server.use(http.get(ME_URL, () => HttpResponse.json(me)));
  const router = createMemoryRouter([{ path: '/', element: <ProfileRoute /> }]);
  return render(
    <TestProviders>
      <RouterProvider router={router} />
    </TestProviders>
  );
};

describe('ProfileRoute', () => {
  it('shows account details', async () => {
    renderRoute({ ...mockMe, groups: ['admins'], auth_method: 'web_session' });
    expect(await screen.findByText('test-user')).toBeInTheDocument();
    expect(screen.getByText('Browser session')).toBeInTheDocument();
    expect(screen.getByText('admins')).toBeInTheDocument();
  });

  it('saves only changed profile fields', async () => {
    const user = userEvent.setup();
    let body: ProfileUpdate | undefined;
    server.use(
      http.patch(`${ME_URL}/profile`, async ({ request }) => {
        body = (await request.json()) as ProfileUpdate;
        return HttpResponse.json({
          ...mockMe,
          profile: { ...mockMe.profile, display_name: 'Ada', display_name_override: 'Ada' },
        });
      })
    );
    renderRoute();

    const save = await screen.findByRole('button', { name: 'Save changes' });
    expect(save).toBeDisabled();
    await user.type(screen.getByRole('textbox', { name: 'Display name' }), 'Ada');
    await user.click(save);

    await waitFor(() => expect(body).toEqual({ display_name: 'Ada' }));
    await waitFor(() => expect(save).toBeDisabled());
  });

  it('is read-only when the profile cannot be edited', async () => {
    renderRoute({ ...mockMe, can_edit_profile: false });
    expect(await screen.findByText('This session cannot change your profile.')).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Display name' })).toBeDisabled();
    expect(screen.queryByRole('button', { name: 'Save changes' })).not.toBeInTheDocument();
  });
});
