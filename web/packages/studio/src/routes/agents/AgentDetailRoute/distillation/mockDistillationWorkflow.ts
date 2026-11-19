// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type MockStageStatus = 'pending' | 'active' | 'completed' | 'failed';

export interface MockStageMetric {
  label: string;
  value: string;
}

export interface MockStage {
  id: 'prepare' | 'customization' | 'eval-original' | 'eval-base' | 'eval-distilled';
  title: string;
  description: string;
  status: MockStageStatus;
  progress: number;
  /** The platform job the stage runs as, once the backend workflow reports it. */
  jobRef?: { kind: string; name: string };
  metrics: MockStageMetric[];
}
