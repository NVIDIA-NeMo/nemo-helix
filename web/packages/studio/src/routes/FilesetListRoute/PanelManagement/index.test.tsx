// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTES } from '@studio/constants/routes';
import { PanelManagement } from '@studio/routes/FilesetListRoute/PanelManagement';
import { renderRoute } from '@studio/tests/util/render';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Outlet, useLocation } from 'react-router';

vi.mock('@nemo/sdk/generated/platform/files', () => ({
  useFilesListFilesetFiles: () => ({ data: undefined, isPending: false, isFetching: false }),
}));

vi.mock('@studio/components/DatasetFileManagementSidePanel', () => ({
  DatasetFileManagementSidePanel: ({
    onClose,
    onFileSelect,
  }: {
    onClose: () => void;
    onFileSelect: (path: string) => void;
  }) => (
    <div>
      <button onClick={onClose}>close fileset panel</button>
      <button onClick={() => onFileSelect('dir/file.jsonl')}>open file</button>
    </div>
  ),
}));

vi.mock('@studio/components/FilesetFilePreviewPanel', () => ({
  FilesetFilePreviewPanel: ({
    onOutsideClick,
    onRenameSuccess,
  }: {
    onOutsideClick: () => void;
    onRenameSuccess: (path: string) => void;
  }) => (
    <div>
      <button onClick={onOutsideClick}>click outside file panel</button>
      <button onClick={() => onRenameSuccess('dir/renamed.jsonl')}>rename file</button>
    </div>
  ),
}));

const LocationProbe = () => {
  const { pathname, search } = useLocation();
  return <div data-testid="location">{`${pathname}${search}`}</div>;
};

const LIST_QUERY = 'page=3&page_size=10&s=compass&sort=name&filters=x';
const FILESET = 'default%2Fmy-fileset';

const renderPanels = (url: string) =>
  renderRoute(undefined, {
    history: url,
    routes: [
      {
        path: ROUTES.workspace.filesets,
        element: (
          <>
            <LocationProbe />
            <Outlet />
            <PanelManagement workspace="default" />
          </>
        ),
        children: [
          { path: ROUTES.workspace.filesetDetails, element: <></> },
          { path: ROUTES.workspace.filesetFile, element: <></> },
        ],
      },
    ],
  });

const expectLocation = (expected: string) =>
  waitFor(() => expect(screen.getByTestId('location').textContent).toBe(expected));

describe('PanelManagement', () => {
  it('keeps the list query when the fileset panel closes', async () => {
    renderPanels(`/workspaces/default/filesets/${FILESET}?${LIST_QUERY}&filesetFolder=dir`);

    await userEvent.click(screen.getByRole('button', { name: 'close fileset panel' }));

    await expectLocation(`/workspaces/default/filesets?${LIST_QUERY}`);
  });

  it('keeps the list query when a file preview opens', async () => {
    renderPanels(`/workspaces/default/filesets/${FILESET}?${LIST_QUERY}&filesetFolder=dir`);

    await userEvent.click(screen.getByRole('button', { name: 'open file' }));

    await expectLocation(
      `/workspaces/default/filesets/${FILESET}/file/dir%2Ffile.jsonl?${LIST_QUERY}`
    );
  });

  it('keeps the list query when the file panel closes from an outside click', async () => {
    renderPanels(`/workspaces/default/filesets/${FILESET}/file/dir%2Ffile.jsonl?${LIST_QUERY}`);

    await userEvent.click(screen.getByRole('button', { name: 'click outside file panel' }));

    await expectLocation(`/workspaces/default/filesets?${LIST_QUERY}`);
  });

  it('keeps the list query after a file rename', async () => {
    renderPanels(`/workspaces/default/filesets/${FILESET}/file/dir%2Ffile.jsonl?${LIST_QUERY}`);

    await userEvent.click(screen.getByRole('button', { name: 'rename file' }));

    await expectLocation(
      `/workspaces/default/filesets/${FILESET}/file/dir%2Frenamed.jsonl?${LIST_QUERY}`
    );
  });

  it('navigates to the bare list route when there is no list query', async () => {
    renderPanels(`/workspaces/default/filesets/${FILESET}`);

    await userEvent.click(screen.getByRole('button', { name: 'close fileset panel' }));

    await expectLocation('/workspaces/default/filesets');
  });
});
