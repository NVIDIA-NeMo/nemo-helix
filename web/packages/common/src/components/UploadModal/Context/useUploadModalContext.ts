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
import type {
  UploadModalState,
  UploadModalAction,
} from '@nemo/common/src/components/UploadModal/Context/useUploadModalReducer';
import { createContext, type Dispatch, useContext } from 'react';

export type UploadModalContextType = [UploadModalState, Dispatch<UploadModalAction>];
export const UploadModalContext = createContext<UploadModalContextType | null>(null);

export class MissingUploadModalProviderError extends Error {
  constructor() {
    super(
      'UploadModalContext not found, ensure an UploadModalProvider exists in the component tree.'
    );
  }
}

/**
 * Allows any component to read/update the state of the UploadModal.
 * State is stored in an UploadModalContext somewhere above the component using this hook.
 *
 * @throws MissingUploadModalProviderError when this hook is used in a component without
 * an UploadModalProvider higher in the component tree.
 */
export const useUploadModalContext = () => {
  const uploadModalContext = useContext(UploadModalContext);

  if (!uploadModalContext) {
    throw new MissingUploadModalProviderError();
  }

  return uploadModalContext;
};
