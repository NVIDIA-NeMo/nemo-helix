// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Tab ids for the model detail page, used as tab values and as the `?tab=<id>` URL value. */
export enum ModelDetailTab {
  Details = 'details',
  Chat = 'chat',
  Training = 'training',
  Logs = 'logs',
}

export const MODEL_DETAIL_DEFAULT_TAB = ModelDetailTab.Details;

export const isModelDetailTab = (value: string | undefined): value is ModelDetailTab =>
  Object.values(ModelDetailTab).includes(value as ModelDetailTab);
