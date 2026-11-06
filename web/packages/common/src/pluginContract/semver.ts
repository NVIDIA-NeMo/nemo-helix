// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const NUM = '(0|[1-9]\\d*)';
const VERSION = new RegExp(
  `^[\\^~]?${NUM}\\.${NUM}\\.${NUM}(?:-[0-9A-Za-z.-]+)?(?:\\+[0-9A-Za-z.-]+)?$`
);

/**
 * Parse `major.minor.patch` (no leading zeros) with an optional leading `^`/`~` and
 * optional `-prerelease` / `+build` suffixes. Null when malformed or beyond safe integers.
 */
export const parseVersion = (v: string): [number, number, number] | null => {
  const m = VERSION.exec(v.trim());
  if (!m) return null;
  const parts = [Number(m[1]), Number(m[2]), Number(m[3])];
  return parts.every(Number.isSafeInteger) ? [parts[0], parts[1], parts[2]] : null;
};
