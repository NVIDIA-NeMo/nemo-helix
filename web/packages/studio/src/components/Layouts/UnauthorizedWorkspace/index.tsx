// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { Lock } from 'lucide-react';

export const UnauthorizedWorkspace = () => (
  <ErrorMessage
    header="You don't have access to this workspace"
    message="You don't have permission to view this workspace. Contact the workspace owner to request access."
    slotMedia={<Lock className="size-16 stroke-2" />}
  />
);
