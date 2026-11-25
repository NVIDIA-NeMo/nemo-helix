// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PluginToast } from '@agent-hardener/host';
import { getApiErrorMessage, toError } from '@nemo/common';

/** "<label>: <reason>", where the reason is the backend's `detail` whenever the API sent one. */
export const describeApiError = (label: string, error: unknown): string =>
  `${label}: ${getApiErrorMessage(toError(error), 'no reason given')}`;

/** An `onError` / `catch` handler that toasts the failure with its backend reason. */
export const toastApiError =
  (toast: PluginToast, label: string) =>
  (error: unknown): void =>
    toast.error(describeApiError(label, error));
