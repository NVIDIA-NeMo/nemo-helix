// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE, SYSTEM_WORKSPACE } from '@nemo/common/src/models/constants';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { WorkspaceSettingsRoute } from '@studio/routes/WorkspaceSettingsRoute';
import { render, screen } from '@studio/tests/util/render';

vi.mock('@studio/hooks/useWorkspaceFromPath');
vi.mock('@studio/routes/WorkspaceSettingsRoute/DeleteWorkspaceModal', () => ({
  DeleteWorkspaceModal: () => null,
}));
vi.mock('@studio/routes/WorkspaceSettingsRoute/EditDescriptionModal', () => ({
  EditDescriptionModal: () => null,
}));

const mockUseWorkspaceFromPath = vi.mocked(useWorkspaceFromPath);

const getDeleteButton = () => screen.getByRole('button', { name: 'Delete Workspace' });

describe('WorkspaceSettingsRoute', () => {
  it('disables the Delete Workspace button for the default workspace', () => {
    mockUseWorkspaceFromPath.mockReturnValue(DEFAULT_WORKSPACE);
    render(<WorkspaceSettingsRoute />);

    expect(getDeleteButton()).toBeDisabled();
    expect(screen.getByText('The default workspace cannot be deleted.')).toBeInTheDocument();
  });

  it('disables the Delete Workspace button for the system workspace', () => {
    mockUseWorkspaceFromPath.mockReturnValue(SYSTEM_WORKSPACE);
    render(<WorkspaceSettingsRoute />);

    expect(getDeleteButton()).toBeDisabled();
    expect(screen.getByText('The system workspace cannot be deleted.')).toBeInTheDocument();
  });

  it('enables the Delete Workspace button for non-default workspaces', () => {
    mockUseWorkspaceFromPath.mockReturnValue('my-workspace');
    render(<WorkspaceSettingsRoute />);

    expect(getDeleteButton()).toBeEnabled();
  });
});
