// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ConfirmationModal } from '@nemo/common/src/components/ConfirmationModal';
import type { FormModalProps } from '@nemo/common/src/components/FormModal';
import type { NotifyFn } from '@nemo/common/src/providers/toast/types';
import type { FC } from 'react';

interface DeleteModalProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  onDelete: () => boolean | Promise<boolean>;
  title: string;
  description?: string;
  confirmationText?: string;
  simpleConfirm?: boolean;
  successText?: string;
  errorText?: string;
  suppressResultToasts?: boolean;
  /** Where result messages go. Defaults to the surrounding ToastProvider; plugins pass `host.notifications.notify`. */
  onNotify?: NotifyFn;
}

export const DeleteConfirmationModal: FC<DeleteModalProps> = ({
  onDelete,
  description,
  confirmationText,
  simpleConfirm = true,
  successText = 'Successfully deleted!',
  errorText = 'Something went wrong. Please try again.',
  suppressResultToasts,
  ...rest
}) => {
  const confirmationDescription =
    description ??
    (simpleConfirm
      ? `Are you sure you want to delete this?`
      : `If you are certain you want to delete this, please type "${confirmationText}" below and click the delete button.`);

  return (
    <ConfirmationModal
      {...rest}
      description={confirmationDescription}
      onConfirm={onDelete}
      submitButtonText="Delete"
      submitButtonColor="danger"
      confirmationText={confirmationText}
      simpleConfirm={simpleConfirm}
      successText={successText}
      errorText={errorText}
      suppressResultToasts={suppressResultToasts}
    />
  );
};
