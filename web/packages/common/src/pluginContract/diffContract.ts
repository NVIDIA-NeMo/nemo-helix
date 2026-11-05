// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* eslint-disable unused-imports/no-unused-vars -- scaffold stubs; remove with the implementation */
import type {
  ContractIssue,
  PluginContract,
  PluginSidecar,
} from '@nemo/common/src/pluginContract/types';

/**
 * Compare a plugin's sidecar against the contract. `studioVersions` maps each
 * shared package to the version Studio ships. A `null` sidecar means the plugin
 * was built without the Studio plugin build helper.
 */
export const diffContract = (
  _sidecar: PluginSidecar | null,
  _contract: PluginContract,
  _studioVersions: Readonly<Record<string, string>>
): ContractIssue[] => {
  throw new Error('not implemented');
};

/** True when any issue should stop the plugin from loading. */
export const hasBlockingIssues = (_issues: readonly ContractIssue[]): boolean => {
  throw new Error('not implemented');
};
