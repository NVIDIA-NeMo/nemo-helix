// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  WorkspaceModelSelectCore,
  type WorkspaceModelSelectCoreProps,
} from '@nemo/common/src/components/ModelSelectV2/WorkspaceModelSelectCore';
import { modelsListModels } from '@nemo/sdk/generated/platform/models';
import type { FC } from 'react';

export type WorkspaceModelSelectProps = Omit<WorkspaceModelSelectCoreProps, 'listModels'>;

/** `WorkspaceModelSelectCore` on Studio's SDK client. */
export const WorkspaceModelSelect: FC<WorkspaceModelSelectProps> = (props) => (
  <WorkspaceModelSelectCore {...props} listModels={modelsListModels} />
);
