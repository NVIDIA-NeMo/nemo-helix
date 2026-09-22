// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';

export interface BulkDeleteModalProps<T> {
  /** Items to delete. */
  items: T[];
  /** Whether the modal is open. */
  open: boolean;
  /**
   * Called when the user confirms. Should perform all deletions and throw on
   * failure — the confirmation modal surfaces the thrown message to the user.
   */
  onDelete: (items: T[]) => Promise<void>;
  /**
   * Modal title. Pass a function to derive it from the count, e.g.
   *   (count) => `Delete ${count} Job${count !== 1 ? 's' : ''}`
   */
  title: string | ((count: number) => string);
  /** Called on both successful delete AND user cancel. */
  onClose: () => void;
}

export const BulkDeleteModal = <T,>({
  items,
  open,
  onDelete,
  title,
  onClose,
}: BulkDeleteModalProps<T>) => {
  const resolvedTitle = typeof title === 'function' ? title(items.length) : title;

  const handleDelete = async (): Promise<boolean> => {
    await onDelete(items);
    onClose();
    return true;
  };

  if (!open) return null;

  return (
    <DeleteConfirmationModal
      open={open}
      onDelete={handleDelete}
      simpleConfirm
      title={resolvedTitle}
      onClose={onClose}
    />
  );
};
