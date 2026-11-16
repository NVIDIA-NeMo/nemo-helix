// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const getDeleteFilesetsDescription = (entity: string, count: number) =>
  count === 1
    ? `The ${entity} and its files are removed from this workspace and can't be restored.`
    : `The ${entity}s and their files are removed from this workspace and can't be restored.`;

export const getDeleteFilesDescription = (count: number) =>
  count === 1
    ? "The file is removed from this fileset and can't be restored."
    : "The files are removed from this fileset and can't be restored.";

export const getDeleteFoldersDescription = (count: number) =>
  count === 1
    ? "The folder and every file in it are removed from this fileset and can't be restored."
    : "The folders and every file in them are removed from this fileset and can't be restored.";

export const DELETE_FILES_AND_FOLDERS_DESCRIPTION =
  "The files and folders, including everything inside the folders, are removed from this fileset and can't be restored.";
