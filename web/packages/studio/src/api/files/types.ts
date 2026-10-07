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
