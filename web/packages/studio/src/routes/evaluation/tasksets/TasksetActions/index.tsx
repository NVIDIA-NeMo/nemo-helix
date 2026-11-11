// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QuickActionsMenuRoot } from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
import type { Taskset } from '@nemo/sdk/generated/evals/schema';
import type { TasksetActionRequest } from '@studio/routes/evaluation/tasksets/types';
import type { FC } from 'react';

interface Props {
  taskset: Taskset;
  onRequest: (request: TasksetActionRequest) => void;
}

/** A taskset's actions menu, shared by the list rows and the detail header. Only reports the
 *  pick — the owner renders `TasksetActionModals`, outside any clickable row. */
export const TasksetActionsMenu: FC<Props> = ({ taskset, onRequest }) => (
  <QuickActionsMenuRoot
    actions={[{ label: 'Edit', onSelect: () => onRequest({ action: 'edit', taskset }) }]}
    deleteAction={{ onSelect: () => onRequest({ action: 'delete', taskset }) }}
  />
);
