// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  filesDownloadFile,
  filesListFilesetFiles,
  filesUploadFile,
} from '@nemo/sdk/generated/platform/files';
import type { FilesetEntry, FilesetLocation } from '@studio/api/files/types';

// One request per file, so a 500-file upload is 500 round trips. Run a bounded number at
// once: unbounded Promise.all would queue them all against the browser's per-host limit
// and lose the first error behind hundreds of in-flight requests.
const UPLOAD_CONCURRENCY = 6;

const forEachBounded = async <T>(
  items: readonly T[],
  task: (item: T) => Promise<void>
): Promise<void> => {
  const queue = [...items];
  const worker = async (): Promise<void> => {
    for (let item = queue.shift(); item !== undefined; item = queue.shift()) {
      try {
        await task(item);
      } catch (error) {
        // Stop the other workers; the caller rolls the fileset back on this error.
        queue.length = 0;
        throw error;
      }
    }
  };

  const results = await Promise.allSettled(
    Array.from({ length: Math.min(UPLOAD_CONCURRENCY, items.length) }, () => worker())
  );
  const failure = results.find((result) => result.status === 'rejected');
  if (failure) throw failure.reason;
};

const asOctetStream = async (data: Blob): Promise<Blob> =>
  new Blob([await data.arrayBuffer()], { type: 'application/octet-stream' });

export const uploadFilesetEntries = (
  workspace: string,
  filesetName: string,
  entries: readonly FilesetEntry[]
): Promise<void> =>
  forEachBounded(entries, async (entry) => {
    await filesUploadFile(workspace, filesetName, entry.path, await asOctetStream(entry.file));
  });

export const copyFilesetFiles = async (
  source: FilesetLocation,
  target: FilesetLocation
): Promise<void> => {
  const { data: files } = await filesListFilesetFiles(source.workspace, source.name);
  await forEachBounded(files, async ({ path }) => {
    const data = await filesDownloadFile(source.workspace, source.name, path);
    await filesUploadFile(target.workspace, target.name, path, await asOctetStream(data));
  });
};
