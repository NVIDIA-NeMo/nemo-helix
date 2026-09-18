// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { toError } from '@nemo/common/src/utils/logger';
import { filesDownloadFile } from '@nemo/sdk/generated/platform/files';
import axios from 'axios';

export interface LargeFileWorkerMessage {
  dataset: string;
  workspace?: string;
  action: 'downloadAsFile';
  path: string;
  /** Configured OIDC bearer passed from the main thread (localStorage is unavailable in workers). */
  accessToken?: string;
}

/**
 * Downloads a file from the NeMo Files service as an ArrayBuffer.
 * Goes through the SDK so the request URL is bound to the configured API base,
 * not whatever URL the caller passes in.
 */
self.onmessage = async function (e: MessageEvent<LargeFileWorkerMessage>) {
  if (e.origin !== '' && e.origin !== self.location.origin) {
    return;
  }

  const { dataset, workspace, action, path, accessToken } = e.data;

  if (accessToken) {
    axios.defaults.headers.common['Authorization'] = `Bearer ${accessToken}`;
  }

  if (action !== 'downloadAsFile') {
    self.postMessage({ done: true, error: `Invalid action: ${action}` });
    return;
  }

  if (!path) {
    self.postMessage({ done: true, error: 'Path is required' });
    return;
  }

  try {
    const response = await filesDownloadFile(workspace || DEFAULT_WORKSPACE, dataset, path);
    const arrayBuffer = await response.arrayBuffer();
    self.postMessage({ done: true, arrayBuffer }, { transfer: [arrayBuffer] });
  } catch (error) {
    self.postMessage({ done: true, error: toError(error).message });
  }
};
