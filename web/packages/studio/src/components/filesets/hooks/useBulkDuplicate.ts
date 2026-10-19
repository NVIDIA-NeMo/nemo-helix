// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { logger } from '@nemo/common/src/utils/logger';
import { filesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { useDatasetFilesUpload } from '@studio/api/datasets/useDatasetFilesUpload';
import { useDownloadFileAsArrayBuffer } from '@studio/components/filesets/hooks/useDownloadFileAsArrayBuffer';
import { FileSystemFile } from '@studio/components/FilesTable/utils';
import { useCallback, useState } from 'react';

export interface UseBulkDuplicateOptions {
  workspace: string;
  datasetName: string;
}

export interface UseBulkDuplicateResult {
  /**
   * Resolves to `true` if every file was uploaded (no download failures, no
   * caught errors). Callers can use this to decide whether to clear the
   * selection — on partial / total failure we keep the selection so the user
   * can retry without re-selecting.
   */
  handleBulkDuplicate: (files: FileSystemFile[]) => Promise<boolean>;
  isDuplicating: boolean;
}

const getDirFromPath = (path: string): string =>
  path.includes('/') ? path.slice(0, path.lastIndexOf('/')) : '';

/**
 * Picks a "-copy" / "-copy-2" / … filename that does not collide with anything
 * in `reserved`. The caller is responsible for adding the returned path back
 * into `reserved` before picking the next sibling, so duplicates of the same
 * source file in a single batch get distinct suffixes.
 */
function pickUniqueCopyPath(originalPath: string, reserved: Set<string>): string {
  const slash = originalPath.lastIndexOf('/');
  const dir = slash >= 0 ? originalPath.slice(0, slash + 1) : '';
  const name = slash >= 0 ? originalPath.slice(slash + 1) : originalPath;
  const dot = name.lastIndexOf('.');
  const base = dot > 0 ? name.slice(0, dot) : name;
  const ext = dot > 0 ? name.slice(dot) : '';

  for (let suffix = 1; ; suffix += 1) {
    const newName = suffix === 1 ? `${base}-copy${ext}` : `${base}-copy-${suffix}${ext}`;
    const candidate = `${dir}${newName}`;
    if (!reserved.has(candidate)) return candidate;
  }
}

/**
 * Duplicates multiple files, picking a non-colliding destination path for each
 * one. The hook:
 *
 *   1. Groups sources by directory and lists each unique directory once via
 *      `filesListFilesetFiles` (prefix-filtered) to see what names are taken.
 *   2. Picks `<name>-copy.<ext>`, falling back to `<name>-copy-2.<ext>`, … for
 *      each source, mutating an in-memory reserved set between picks so two
 *      sibling duplicates in the same batch don't both land on `-copy`.
 *   3. Downloads source bytes in parallel via {@link useDownloadFileAsArrayBuffer}
 *      and uploads the renamed blobs as a single batch.
 *
 * Note: this is still a best-effort check against an eventually-consistent
 * listing — a concurrent writer between step 1 and step 3 can still lose to
 * our PUT. Single-client usage is the intended case.
 */
export function useBulkDuplicate(options: UseBulkDuplicateOptions): UseBulkDuplicateResult {
  const { workspace, datasetName } = options;
  const toast = useToast();
  const downloadAsArrayBuffer = useDownloadFileAsArrayBuffer();
  const { mutateAsync: uploadFiles } = useDatasetFilesUpload();
  // Tracks listing + download + upload phases so callers can disable the
  // trigger for the full flow, not just the upload mutation.
  const [isDuplicating, setIsDuplicating] = useState(false);

  const handleBulkDuplicate = useCallback(
    async (files: FileSystemFile[]): Promise<boolean> => {
      if (files.length === 0) return true;

      const toastId = toast.workingWithId(
        files.length === 1 ? 'Duplicating file...' : `Duplicating ${files.length} files...`
      );

      setIsDuplicating(true);
      try {
        // List each unique source directory once and collect the set of
        // already-taken paths under it. Using the `path` prefix keeps payloads
        // bounded to the relevant subtree.
        const uniqueDirs = Array.from(new Set(files.map((f) => getDirFromPath(f.path))));
        const reservedByDir = new Map<string, Set<string>>();
        await Promise.all(
          uniqueDirs.map(async (dir) => {
            const response = await filesListFilesetFiles(workspace, datasetName, {
              path: dir ? `${dir}/` : undefined,
            });
            reservedByDir.set(dir, new Set(response.data.map((f) => f.path)));
          })
        );

        // Reserve a destination for each source in input order, mutating
        // `reserved` between picks so sibling duplicates get `-copy`, `-copy-2`,
        // etc. instead of colliding on the same `-copy` name.
        const plans = files.map((file) => {
          const dir = getDirFromPath(file.path);
          const reserved = reservedByDir.get(dir) ?? new Set<string>();
          const newPath = pickUniqueCopyPath(file.path, reserved);
          reserved.add(newPath);
          reservedByDir.set(dir, reserved);
          return { sourcePath: file.path, newPath };
        });

        const buffers = await Promise.all(
          plans.map((p) => downloadAsArrayBuffer({ workspace, datasetName, path: p.sourcePath }))
        );

        const newFiles: File[] = [];
        const failures: string[] = [];
        plans.forEach((plan, i) => {
          const buffer = buffers[i];
          if (!buffer) {
            failures.push(plan.sourcePath);
            return;
          }
          const blob = new Blob([buffer], { type: 'application/octet-stream' });
          newFiles.push(new File([blob], plan.newPath, { type: blob.type }));
        });

        if (newFiles.length > 0) {
          await uploadFiles({ workspace, datasetName, files: newFiles });
        }

        toast.dismissToast(toastId);

        if (failures.length === 0) {
          toast.success(
            newFiles.length === 1
              ? 'File duplicated successfully'
              : `Duplicated ${newFiles.length} files`
          );
          return true;
        } else if (newFiles.length === 0) {
          toast.error(
            files.length === 1 ? 'Failed to duplicate file' : 'Failed to duplicate files'
          );
          return false;
        } else {
          toast.error(
            `Duplicated ${newFiles.length} of ${plans.length} files. ${failures.length} failed.`
          );
          return false;
        }
      } catch (err) {
        logger.error('Bulk duplicate failed', err);
        toast.dismissToast(toastId);
        toast.error(files.length === 1 ? 'Failed to duplicate file' : 'Failed to duplicate files');
        return false;
      } finally {
        setIsDuplicating(false);
      }
    },
    [workspace, datasetName, downloadAsArrayBuffer, uploadFiles, toast]
  );

  return { handleBulkDuplicate, isDuplicating };
}
