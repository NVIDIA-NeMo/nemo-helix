// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { nameTakenMessage } from '@studio/components/form/entityNameCheck';

export const filesetNameTakenMessage = (name: string, workspace: string): string =>
  nameTakenMessage('fileset', name, workspace);
