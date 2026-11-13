// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ConfirmationModal } from '@nemo/common/src/components/ConfirmationModal';
import { DeleteIdList } from '@nemo/common/src/components/DeleteConfirmationModal/DeleteIdList';
import type { FormModalProps } from '@nemo/common/src/components/FormModal';
import type { NotifyFn } from '@nemo/common/src/providers/toast/types';
import type { FC } from 'react';

export type DeleteVerb = 'delete' | 'remove';

const VERB_COPY: Record<DeleteVerb, { action: string; past: string }> = {
  delete: { action: 'Delete', past: 'Deleted' },
  remove: { action: 'Remove', past: 'Removed' },
};

export interface DeleteModalProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  onDelete: () => boolean | Promise<boolean>;
  title?: string;
  entity?: string;
  entityPlural?: string;
  ids?: readonly string[];
  verb?: DeleteVerb;
  description?: string;
  confirmationText?: string;
  simpleConfirm?: boolean;
  successText?: string;
  errorText?: string;
  suppressResultToasts?: boolean;
  /** Where result messages go. Defaults to the surrounding ToastProvider; plugins pass `host.notifications.notify`. */
  onNotify?: NotifyFn;
}

const describeTarget = (entity: string, entityPlural: string, ids: readonly string[]) => {
  const [firstId] = ids;
  const isBulk = ids.length > 1;
  const isSingle = !isBulk && firstId !== undefined;
  const countTarget = `${ids.length} ${entityPlural}`;
  return {
    isBulk,
    target: isSingle ? `${entity} '${firstId}'` : countTarget,
    buttonTarget: isSingle ? entity : countTarget,
  };
};

export const DeleteConfirmationModal: FC<DeleteModalProps> = ({
  onDelete,
  title,
  entity,
  entityPlural = `${entity}s`,
  ids = [],
  verb = 'delete',
  description,
  confirmationText,
  simpleConfirm = true,
  successText,
  errorText,
  suppressResultToasts,
  ...rest
}) => {
  const { action, past } = VERB_COPY[verb];
  const named = entity ? describeTarget(entity, entityPlural, ids) : undefined;

  const confirmationDescription =
    description ??
    (simpleConfirm
      ? `Are you sure you want to ${verb} this?`
      : `If you are certain you want to ${verb} this, please type "${confirmationText}" below and click the ${verb} button.`);

  return (
    <ConfirmationModal
      {...rest}
      title={title ?? (named ? `${action} ${named.target}?` : action)}
      description={confirmationDescription}
      onConfirm={onDelete}
      submitButtonText={named ? `${action} ${named.buttonTarget}` : action}
      submitButtonColor="danger"
      initialFocus={simpleConfirm ? 'cancel' : undefined}
      confirmationText={confirmationText}
      simpleConfirm={simpleConfirm}
      successText={
        successText ?? (named ? `${past} ${named.target}.` : `Successfully ${past.toLowerCase()}!`)
      }
      errorText={
        errorText ??
        (named ? `Couldn't ${verb} ${named.target}.` : 'Something went wrong. Please try again.')
      }
      suppressResultToasts={suppressResultToasts}
    >
      {named?.isBulk && <DeleteIdList ids={ids} />}
    </ConfirmationModal>
  );
};
