// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const getDeleteGuardrailConfigsDescription = (count: number) =>
  count === 1
    ? "The guardrail config is deleted and can't be restored."
    : "The guardrail configs are deleted and can't be restored.";
