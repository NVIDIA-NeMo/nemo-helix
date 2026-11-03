// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { Flex, Text } from '@nvidia/foundations-react-core';
import { FC, useState } from 'react';

export interface CancelJobModalProps {
  open: boolean;
  jobName: string;
  onClose: () => void;
  /** Rejecting keeps the modal open and shows the error inline. Resolving closes it via onClose. */
  onConfirm: () => Promise<void>;
}

export const CancelJobModal: FC<CancelJobModalProps> = ({ open, jobName, onClose, onConfirm }) => {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // FormModal ignores close requests while `disabled`, so no in-flight guard is needed here.
  const handleClose = () => {
    setError(null);
    onClose();
  };

  const handleSubmit = async () => {
    setPending(true);
    setError(null);
    try {
      await onConfirm();
      setPending(false);
      onClose();
    } catch (e) {
      setPending(false);
      // A rejection is not guaranteed to be an Error (`throw undefined`); getErrorMessage reads `.message`.
      setError(
        e instanceof Error ? getErrorMessage(e, 'Failed to cancel job') : 'Failed to cancel job'
      );
    }
  };

  return (
    <FormModal
      open={open}
      title={`Cancel ${jobName}`}
      submitButtonText="Cancel Job"
      errorText={error}
      onSubmit={(e) => {
        e.preventDefault();
        handleSubmit();
      }}
      onClose={handleClose}
      disabled={pending}
      loading={pending}
      attributes={{
        SubmitButton: { color: 'danger', 'aria-label': 'Cancel Job' },
        CancelButton: { autoFocus: true },
      }}
    >
      <Flex>
        <Text className="leading-relaxed">
          Cancelling stops this job and can&apos;t be undone. A cancelled job can&apos;t be resumed
          or rerun, but it can still be deleted. It may show as Cancelling briefly before it stops.
        </Text>
      </Flex>
    </FormModal>
  );
};
