/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { UploadModalContext } from '@nemo/common/src/components/UploadModal/Context/useUploadModalContext';
import {
  type UploadModalState,
  useUploadModalReducer,
} from '@nemo/common/src/components/UploadModal/Context/useUploadModalReducer';
import type { FC, PropsWithChildren } from 'react';

interface Props {
  initialState?: UploadModalState;
}

/**
 * Provides a shared state for the UploadModal component.
 */
export const UploadModalProvider: FC<PropsWithChildren<Props>> = ({ initialState, children }) => {
  const [state, dispatch] = useUploadModalReducer(initialState);

  return (
    <UploadModalContext.Provider value={[state, dispatch]}>{children}</UploadModalContext.Provider>
  );
};
