// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Security gate applied before dynamically importing a plugin bundle.
 * Only paths served by the platform under /plugin-ui/ are trusted.
 */

/** Plugin names must start with a letter and contain only lowercase alphanumeric chars and hyphens. */
const VALID_BUNDLE_URL = /^\/plugin-ui\/[a-z][a-z0-9-]+\/(?:[a-f0-9]{64}\/)?index\.js$/;

export function isTrustedBundleUrl(url: string): boolean {
  return VALID_BUNDLE_URL.test(url);
}
