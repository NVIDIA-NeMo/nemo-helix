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

import type { ButtonProps } from '@nvidia/foundations-react-core';
import { CreateButton } from '@studio/components/common/CreateButton';
import { NewDatasetPanel } from '@studio/components/NewDatasetButton/NewDatasetPanel';
import { useBoolean } from '@studio/util/hooks/useBoolean';
import type { FC } from 'react';

interface Props extends Omit<ButtonProps, 'onClick'> {
  expanded?: boolean;
  withIcon?: boolean;
}

export const NewDatasetButton: FC<Props> = ({ ...buttonProps }) => {
  const [isOpen, setOpen, setClosed] = useBoolean(false);

  return (
    <>
      <CreateButton onClick={() => setOpen()} {...buttonProps}>
        Create Dataset
      </CreateButton>
      {isOpen && <NewDatasetPanel open={isOpen} onClose={setClosed} />}
    </>
  );
};
