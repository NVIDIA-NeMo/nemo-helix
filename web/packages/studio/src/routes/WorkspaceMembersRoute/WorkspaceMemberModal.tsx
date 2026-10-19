/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { zodResolver } from '@hookform/resolvers/zod';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { RadioCard } from '@nemo/common/src/components/RadioCard';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { handleFormErrorsGeneric } from '@nemo/common/src/utils/forms/error';
import {
  getEntitiesListWorkspaceMembersQueryKey,
  useEntitiesAddWorkspaceMember,
  useEntitiesUpdateWorkspaceMember,
} from '@nemo/sdk/generated/platform/entity-store';
import type { WorkspaceMember } from '@nemo/sdk/generated/platform/schema';
import { FormField, RadioGroupRoot, Stack } from '@nvidia/foundations-react-core';
import { queryClient } from '@studio/api/queryClient';
import {
  primaryWorkspaceMemberRole,
  WORKSPACE_MEMBER_ROLES,
  WORKSPACE_ROLE_DESCRIPTIONS,
  type WorkspaceMemberRole,
} from '@studio/routes/WorkspaceMembersRoute/workspaceMemberRoles';
import { FC, useEffect, useMemo } from 'react';
import { Controller, SubmitHandler, useForm } from 'react-hook-form';
import { z } from 'zod';

export type WorkspaceMemberModalMode = 'add' | 'edit';

interface WorkspaceMemberFormValues {
  principal: string;
  role: WorkspaceMemberRole;
}

interface WorkspaceMemberModalProps {
  open: boolean;
  onClose: () => void;
  workspace: string;
  mode: WorkspaceMemberModalMode;
  member?: WorkspaceMember | null;
  existingMembers?: WorkspaceMember[];
}

function createWorkspaceMemberFormSchema(
  mode: WorkspaceMemberModalMode,
  existingPrincipals: string[]
) {
  return z
    .object({
      principal: z.string(),
      role: z.enum(['Viewer', 'Editor', 'Admin']),
    })
    .superRefine((data, ctx) => {
      if (mode === 'add' && data.principal.trim() === '') {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'Enter an email or user ID.',
          path: ['principal'],
        });
      }
      if (mode === 'add' && existingPrincipals.includes(data.principal.trim())) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'User already exists.',
          path: ['principal'],
        });
      }
    });
}

export const WorkspaceMemberModal: FC<WorkspaceMemberModalProps> = ({
  open,
  onClose,
  workspace,
  mode,
  member,
  existingMembers = [],
}) => {
  const toast = useToast();
  const isAdd = mode === 'add';

  const existingPrincipals = useMemo(
    () => existingMembers.map((m) => m.principal),
    [existingMembers]
  );

  const schema = useMemo(
    () => createWorkspaceMemberFormSchema(mode, existingPrincipals),
    [mode, existingPrincipals]
  );

  const {
    control,
    handleSubmit,
    reset,
    setError,
    clearErrors,
    getValues,
    formState: { isSubmitting },
  } = useForm<WorkspaceMemberFormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      principal: '',
      role: 'Viewer',
    },
    mode: 'onSubmit',
    reValidateMode: 'onChange',
    shouldFocusError: true,
  });

  const { mutateAsync: addMember } = useEntitiesAddWorkspaceMember();
  const { mutateAsync: updateMember } = useEntitiesUpdateWorkspaceMember();

  useEffect(() => {
    if (!open) {
      return;
    }
    clearErrors();
    reset({
      principal: mode === 'edit' && member ? member.principal : '',
      role: mode === 'edit' && member ? primaryWorkspaceMemberRole(member.roles) : 'Viewer',
    });
  }, [open, mode, member, reset, clearErrors]);

  const title = isAdd ? 'Add a Workspace Member' : 'Edit Workspace Member';
  const instruction = isAdd
    ? 'Invite members to a workspace and assign them a role.'
    : "Update this member's role within this workspace.";

  const onValid: SubmitHandler<WorkspaceMemberFormValues> = async (data) => {
    clearErrors();
    const rolesList = [data.role];

    try {
      if (isAdd) {
        /** Disabled fields are omitted from submit `data`; use full form values for principal. */
        const p = getValues('principal').trim();
        await addMember({
          workspace,
          data: { principal: p, roles: rolesList },
        });
        toast.success(`Added ${p} to the workspace.`);
      } else if (member) {
        await updateMember({
          workspace,
          principalId: member.principal,
          data: { roles: rolesList },
        });
        toast.success('Roles updated.');
      }
      await queryClient.invalidateQueries({
        queryKey: getEntitiesListWorkspaceMembersQueryKey(workspace),
      });
      onClose();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Request failed.';
      setError('principal', { message, type: 'server' });
      toast.error(message);
    }
  };

  const formDisabled = isSubmitting;

  return (
    <FormModal
      className="w-[600px]"
      open={open}
      title={title}
      instruction={instruction}
      submitButtonText={isAdd ? 'Add Member' : 'Save'}
      disabled={formDisabled}
      loading={formDisabled}
      onSubmit={handleSubmit(onValid, handleFormErrorsGeneric({ title: 'Workspace member form' }))}
      onClose={onClose}
    >
      <Stack className="my-4" gap="6">
        <ControlledTextInput
          useControllerProps={{
            name: 'principal',
            control,
            disabled: mode === 'edit' || formDisabled,
          }}
          label={isAdd ? 'Email' : 'Member'}
          placeholder={isAdd ? 'user@example.com' : 'Enter member email'}
          attributes={{
            Input: {
              'aria-label': isAdd ? 'Email Address' : 'Member',
              autoComplete: 'off',
            },
          }}
        />
        <Controller
          name="role"
          control={control}
          render={({ field, fieldState }) => (
            <Stack gap="density-sm">
              <FormField
                name="role"
                slotLabel="Role"
                slotError={fieldState.error?.message}
                status={fieldState.error ? 'error' : undefined}
              >
                <RadioGroupRoot
                  name="workspace-member-role"
                  value={field.value}
                  onValueChange={field.onChange}
                  onBlur={field.onBlur}
                  className="w-full"
                  disabled={formDisabled}
                >
                  <Stack gap="3">
                    {WORKSPACE_MEMBER_ROLES.map((role) => (
                      <RadioCard
                        key={role}
                        value={role}
                        label={role}
                        description={WORKSPACE_ROLE_DESCRIPTIONS[role]}
                        labelSide="left"
                      />
                    ))}
                  </Stack>
                </RadioGroupRoot>
              </FormField>
            </Stack>
          )}
        />
      </Stack>
    </FormModal>
  );
};
