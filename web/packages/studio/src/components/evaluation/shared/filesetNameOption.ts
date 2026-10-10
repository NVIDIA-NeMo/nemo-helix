// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** A fileset option keyed by its bare name, which is how the evaluation pickers store it. */
export const filesetNameOption = (fileset: { name: string }) => ({
  value: fileset.name,
  label: fileset.name,
});
