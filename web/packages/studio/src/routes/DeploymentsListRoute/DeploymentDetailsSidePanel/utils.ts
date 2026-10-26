// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Engine } from '@nemo/sdk/generated/platform/schema';

export function formatDeploymentEngine(engine: string | undefined): string {
  if (!engine) return '-';
  const normalized = engine.toLowerCase();
  if (normalized === Engine.nim) return 'NIM';
  if (normalized === Engine.vllm) return 'vLLM';
  if (normalized === Engine.generic) return 'Generic';
  return engine;
}
