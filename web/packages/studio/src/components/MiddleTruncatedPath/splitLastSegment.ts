// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const LAST_SEGMENT = /[^/]+\/?$/;

/** Splits a path before its last segment (a trailing slash stays with it). */
export const splitLastSegment = (path: string): [head: string, tail: string] => {
  const tailStart = path.search(LAST_SEGMENT);
  return tailStart === -1 ? [path, ''] : [path.slice(0, tailStart), path.slice(tailStart)];
};
