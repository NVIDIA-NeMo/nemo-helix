// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ConfirmationModal } from '@nemo/common/src/components/ConfirmationModal';
import type { FC } from 'react';

interface Props {
  onClose: () => void;
  onConfirm: () => void;
  description?: string;
}

/**
 * Guards an edited field mapping against a stray dismissal. Mount only while
 * needed: it carries its own form state, which would otherwise settle
 * asynchronously behind the still-open transform modal.
 */
export const DiscardTransformModal: FC<Props> = ({
  onClose,
  onConfirm,
  description = 'Your field mapping has not been submitted. Closing now discards it.',
}) => (
  <ConfirmationModal
    open
    onClose={onClose}
    onConfirm={() => {
      onConfirm();
      return true;
    }}
    title="Discard this transform?"
    description={description}
    submitButtonText="Discard"
    cancelButtonText="Keep editing"
    submitButtonColor="danger"
    suppressResultToasts
  />
);
