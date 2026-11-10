// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Where a run publishes: an optional experiment and an optional evaluation name. */
export interface RunTargetFormFields {
  /** Entity id of the experiment to link up front; empty leaves the run in the agent's default. */
  experimentId: string;
  /** Empty means a name is generated from the agent at submit. */
  evaluationRecordName: string;
}
