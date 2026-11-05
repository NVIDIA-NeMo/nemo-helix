// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Parse `x.y.z` (optional leading `^`/`~`, optional `-pre`/`+build` suffix); null otherwise. */
export const parseVersion = (v: string): [number, number, number] | null => {
  const m = /^[\^~]?(\d+)\.(\d+)\.(\d+)(?:[-+][0-9A-Za-z.+-]*)?$/.exec(v.trim());
  return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : null;
};
