// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface FilesetEntry {
  readonly path: string;
  readonly file: File;
}

export interface FilesetLocation {
  readonly workspace: string;
  readonly name: string;
}

/**
 * Where a bundle of files — a config plus the assets it references — comes from: files the user
 * picked, still to be uploaded, or a fileset that already holds them and is read in place.
 */
export type BundleSource =
  | { readonly entries: readonly FilesetEntry[]; readonly fileset?: never }
  | { readonly fileset: string; readonly entries?: never };
