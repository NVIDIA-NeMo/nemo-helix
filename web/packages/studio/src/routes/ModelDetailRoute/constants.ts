// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QUERY_PARAMETERS } from '@studio/routes/constants';

/** Tab ids for the model detail page, used as tab values and as the `?tab=<id>` URL value. */
export enum ModelDetailTab {
  Details = 'details',
  Chat = 'chat',
  Evaluations = 'evaluations',
  Files = 'files',
  Deployment = 'deployment',
  Training = 'training',
  Logs = 'logs',
}

export type ModelDetailParent = 'fine-tuning' | 'catalog';

// Relative to the current path, so a tab link works under both the catalog and fine-tuning URLs.
export const getModelTabLink = (tab: ModelDetailTab) => `?${QUERY_PARAMETERS.tab}=${tab}`;
