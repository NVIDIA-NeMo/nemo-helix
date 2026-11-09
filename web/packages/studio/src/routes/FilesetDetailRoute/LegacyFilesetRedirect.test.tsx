// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTES } from '@studio/constants/routes';
import { LegacyFilesetRedirect } from '@studio/routes/FilesetDetailRoute/LegacyFilesetRedirect';
import { LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { render, screen } from '@testing-library/react';
import { createMemoryRouter, RouterProvider } from 'react-router';

const renderAt = (url: string) => {
  const router = createMemoryRouter(
    [
      { path: ROUTES.workspace.filesetDetails, element: <LegacyFilesetRedirect /> },
      { path: ROUTES.workspace.filesetFile, element: <LegacyFilesetRedirect /> },
      { path: ROUTES.workspace.filesetDetail, element: <LocationDisplay /> },
    ],
    { initialEntries: [url] }
  );
  render(<RouterProvider router={router} />);
};

describe('LegacyFilesetRedirect', () => {
  it.each([
    ['/workspaces/ws/filesets/my-set', '/workspaces/ws/filesets/my-set/detail'],
    ['/workspaces/ws/filesets/ws%2Fmy-set', '/workspaces/ws/filesets/my-set/detail'],
    [
      '/workspaces/ws/filesets/ws%2Fmy-set?filesetFolder=nested%2Fdir',
      '/workspaces/ws/filesets/my-set/detail?tab=files&filesetFolder=nested%2Fdir',
    ],
    [
      '/workspaces/ws/filesets/ws%2Fmy-set/file/nested%2Fdata.jsonl',
      '/workspaces/ws/filesets/my-set/detail?tab=files&file=nested%2Fdata.jsonl',
    ],
  ])('redirects %s to %s', async (from, to) => {
    renderAt(from);
    expect(await screen.findByTestId(LOCATION_DISPLAY_TEST_ID)).toHaveTextContent(to);
  });
});
