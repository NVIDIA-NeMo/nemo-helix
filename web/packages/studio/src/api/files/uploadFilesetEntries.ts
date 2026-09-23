// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesUploadFile } from '@nemo/sdk/generated/platform/files';

export interface FilesetEntry {
  readonly path: string;
  readonly file: File;
}

// One request per file, so a 500-file upload is 500 round trips. Run a bounded number at
// once: unbounded Promise.all would queue them all against the browser's per-host limit
// and lose the first error behind hundreds of in-flight requests.
const UPLOAD_CONCURRENCY = 6;

export const uploadFilesetEntries = async (
  workspace: string,
  filesetName: string,
  entries: readonly FilesetEntry[]
): Promise<void> => {
  const queue = [...entries];
  const worker = async (): Promise<void> => {
    for (let entry = queue.shift(); entry; entry = queue.shift()) {
      const blob = new Blob([await entry.file.arrayBuffer()], { type: 'application/octet-stream' });
      await filesUploadFile(workspace, filesetName, entry.path, blob);
    }
  };

  await Promise.all(
    Array.from({ length: Math.min(UPLOAD_CONCURRENCY, entries.length) }, () => worker())
  );
};
