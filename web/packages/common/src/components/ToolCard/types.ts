// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface ToolParameter {
  readonly name: string;
  readonly type?: string;
  readonly description?: string;
  readonly required: boolean;
}
