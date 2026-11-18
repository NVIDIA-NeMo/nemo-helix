// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema/FilesetPurpose';
import type { ButtonProps } from '@nvidia/foundations-react-core';
import { CreateButton } from '@studio/components/common/CreateButton';
import { FilesetCreateModal } from '@studio/components/FilesetCreateModal';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBoolean } from '@studio/util/hooks/useBoolean';
import type { FC } from 'react';

interface Props extends Omit<ButtonProps, 'onClick'> {
  expanded?: boolean;
  withIcon?: boolean;
}

export const NewModelFilesetButton: FC<Props> = ({ ...buttonProps }) => {
  const [isOpen, setOpen, setClosed] = useBoolean(false);
  const workspace = useWorkspaceFromPath();

  return (
    <>
      <CreateButton onClick={() => setOpen()} {...buttonProps}>
        Create Model Fileset
      </CreateButton>
      {isOpen && (
        <FilesetCreateModal
          open={isOpen}
          onClose={setClosed}
          workspace={workspace}
          purpose={FilesetPurpose.model}
        />
      )}
    </>
  );
};
