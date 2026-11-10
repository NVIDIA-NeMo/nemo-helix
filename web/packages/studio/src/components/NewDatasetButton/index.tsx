/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema/FilesetPurpose';
import type { ButtonProps } from '@nvidia/foundations-react-core';
import { CreateButton } from '@studio/components/common/CreateButton';
import { DatasetCreateModal } from '@studio/components/DatasetCreateModal';
import { FilesetCreateModal } from '@studio/components/FilesetCreateModal';
import { FILESET_DETAILS_ENABLED } from '@studio/constants/environment';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBoolean } from '@studio/util/hooks/useBoolean';
import type { FC } from 'react';

interface Props extends Omit<ButtonProps, 'onClick'> {
  expanded?: boolean;
  withIcon?: boolean;
}

export const NewDatasetButton: FC<Props> = ({ ...buttonProps }) => {
  const [isOpen, setOpen, setClosed] = useBoolean(false);
  const workspace = useWorkspaceFromPath();

  return (
    <>
      <CreateButton onClick={() => setOpen()} {...buttonProps}>
        Create Dataset
      </CreateButton>
      {isOpen &&
        (FILESET_DETAILS_ENABLED ? (
          <FilesetCreateModal
            open={isOpen}
            onClose={setClosed}
            workspace={workspace}
            purpose={FilesetPurpose.dataset}
          />
        ) : (
          <DatasetCreateModal open={isOpen} onClose={setClosed} onDatasetCreated={setClosed} />
        ))}
    </>
  );
};
