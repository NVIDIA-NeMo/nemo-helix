// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DASHBOARD_QUICKSTART_DISMISSED_KEY_PREFIX } from '@studio/util/localStorage';

export const getQuickstartDismissedKey = (workspace: string): string =>
  `${DASHBOARD_QUICKSTART_DISMISSED_KEY_PREFIX}:${workspace}`;
