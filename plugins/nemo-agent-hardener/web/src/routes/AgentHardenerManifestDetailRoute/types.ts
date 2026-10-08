// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type AttackIntensity = 'light' | 'standard' | 'thorough';

export type ReplaySource = 'last' | 'upload';

export type BenignSource = 'manifest' | 'upload';

export interface DefenderSelection {
  guardrails: boolean;
  openshell: boolean;
}
