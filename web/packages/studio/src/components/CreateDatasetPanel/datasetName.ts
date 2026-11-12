// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export function getDatasetNameError(
  _name: string,
  _opts: { taken: boolean; workspace: string },
): string | undefined {
  throw new Error('not implemented');
}

export function inferDatasetName(_seed: string): string {
  throw new Error('not implemented');
}
