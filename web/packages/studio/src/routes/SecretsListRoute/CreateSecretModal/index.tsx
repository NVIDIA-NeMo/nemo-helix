/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import {
  CreateSecretModal as CreateSecretModalBase,
  type CreateSecretFormData,
  type CreateSecretModalProps as BaseProps,
} from '@nemo/common/src/components/CreateSecretModal';
import {
  getSecretsListSecretsQueryKey,
  useSecretsCreateSecret,
} from '@nemo/sdk/generated/platform/secrets';
import { useQueryClient } from '@tanstack/react-query';
import type { FC } from 'react';

interface CreateSecretModalProps extends Pick<BaseProps, 'open' | 'onClose'> {
  workspace: string;
  /** Optional. After create, called with the new secret name (e.g. create-dataset flow sets the Secret Key field). */
  onSecretCreated?: (secretName: string) => void;
}

export const CreateSecretModal: FC<CreateSecretModalProps> = ({
  workspace,
  open,
  onClose,
  onSecretCreated,
}) => {
  const queryClient = useQueryClient();

  const {
    mutateAsync: createSecret,
    error: createError,
    isPending,
    reset: resetCreateMutation,
  } = useSecretsCreateSecret();

  const handleCreate = async (data: CreateSecretFormData): Promise<void> => {
    const secret = await createSecret({ workspace, data });
    onSecretCreated?.(secret.name);
    queryClient.invalidateQueries({ queryKey: getSecretsListSecretsQueryKey(workspace) });
  };

  const handleClose = () => {
    resetCreateMutation();
    onClose();
  };

  return (
    <CreateSecretModalBase
      open={open}
      onClose={handleClose}
      onCreate={handleCreate}
      pending={isPending}
      errorText={createError ? getErrorMessage(createError) : undefined}
    />
  );
};
