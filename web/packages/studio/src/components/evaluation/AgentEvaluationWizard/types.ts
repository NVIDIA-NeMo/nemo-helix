// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** A dataset already in a fileset of the wizard's workspace. */
export interface WizardDataset {
  fileset: string;
  file: string;
  /** Glob over a folder of Parquet batches read as one dataset; `file` is then its first batch. */
  batchGlob?: string;
}
