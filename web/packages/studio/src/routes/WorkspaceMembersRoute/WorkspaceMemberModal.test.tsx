/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { queryClient } from '@studio/api/queryClient';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';

const { mockAddMember, mockUpdateMember } = vi.hoisted(() => ({
  mockAddMember: vi.fn(),
  mockUpdateMember: vi.fn(),
}));

vi.mock('@nemo/common/src/components/FormModal', () => ({
  FormModal: ({
    children,
    open,
    onSubmit,
    submitButtonText,
    disabled,
  }: {
    children: React.ReactNode;
    open: boolean;
    onSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
    submitButtonText: string;
    disabled?: boolean;
    [key: string]: unknown;
  }) =>
    open ? (
      <form
        data-testid="form-modal"
        onSubmit={(e) => {
          e.preventDefault();
          onSubmit(e);
        }}
      >
        {children}
        <button type="submit" disabled={disabled}>
          {submitButtonText}
        </button>
      </form>
    ) : null,
}));

vi.mock('@nemo/sdk/generated/platform/entity-store', async (importOriginal) => {
  const mod = await importOriginal<typeof import('@nemo/sdk/generated/platform/entity-store')>();
  return {
    ...mod,
    useEntitiesAddWorkspaceMember: () => ({ mutateAsync: mockAddMember }),
    useEntitiesUpdateWorkspaceMember: () => ({ mutateAsync: mockUpdateMember }),
  };
});

const mockMember = {
  principal: 'user@example.com',
  roles: ['Viewer'] as ('Viewer' | 'Editor' | 'Admin')[],
  workspace: 'test-workspace',
};

describe('WorkspaceMemberModal', () => {
  beforeEach(() => {
    mockAddMember.mockReset();
    mockUpdateMember.mockReset();
    mockAddMember.mockResolvedValue(undefined);
    mockUpdateMember.mockResolvedValue(undefined);
    vi.spyOn(queryClient, 'invalidateQueries').mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('edit mode', () => {
    it('labels the principal field as "Member"', async () => {
      const { WorkspaceMemberModal } =
        await import('@studio/routes/WorkspaceMembersRoute/WorkspaceMemberModal');

      render(
        <TestProviders>
          <WorkspaceMemberModal
            open
            onClose={vi.fn()}
            workspace="test-workspace"
            mode="edit"
            member={mockMember}
          />
        </TestProviders>
      );

      expect(screen.getByText('Member')).toBeInTheDocument();
    });

    it('submits role update and closes the modal', async () => {
      const onClose = vi.fn();
      const user = userEvent.setup();
      const { WorkspaceMemberModal } =
        await import('@studio/routes/WorkspaceMembersRoute/WorkspaceMemberModal');

      render(
        <TestProviders>
          <WorkspaceMemberModal
            open
            onClose={onClose}
            workspace="test-workspace"
            mode="edit"
            member={mockMember}
          />
        </TestProviders>
      );

      await user.click(screen.getByRole('radio', { name: /^Editor$/i }));
      await user.click(screen.getByRole('button', { name: 'Save' }));

      await waitFor(() => {
        expect(mockUpdateMember).toHaveBeenCalledWith({
          workspace: 'test-workspace',
          principalId: 'user@example.com',
          data: { roles: ['Editor'] },
        });
      });
      await waitFor(() => {
        expect(queryClient.invalidateQueries).toHaveBeenCalled();
      });
      expect(onClose).toHaveBeenCalledTimes(1);
    });
  });

  describe('add mode', () => {
    it('submits new member and closes the modal', async () => {
      const onClose = vi.fn();
      const user = userEvent.setup();
      const { WorkspaceMemberModal } =
        await import('@studio/routes/WorkspaceMembersRoute/WorkspaceMemberModal');

      render(
        <TestProviders>
          <WorkspaceMemberModal open onClose={onClose} workspace="test-workspace" mode="add" />
        </TestProviders>
      );

      await user.type(screen.getByLabelText('Email Address'), 'new.user@example.com');
      await user.click(screen.getByRole('button', { name: 'Add Member' }));

      await waitFor(() => {
        expect(mockAddMember).toHaveBeenCalledWith({
          workspace: 'test-workspace',
          data: { principal: 'new.user@example.com', roles: ['Viewer'] },
        });
      });
      await waitFor(() => {
        expect(queryClient.invalidateQueries).toHaveBeenCalled();
      });
      expect(onClose).toHaveBeenCalledTimes(1);
    });
  });
});
