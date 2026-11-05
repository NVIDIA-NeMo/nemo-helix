// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Canonical tab IDs for the Fileset Detail page.
 *
 * These string values appear as TabsTrigger/TabsContent `value` props and as
 * the `?tab=<id>` URL query value. Add a tab here, reference it everywhere.
 *
 * The Card tab's label is purpose-dependent (e.g. "Model Card" vs "Dataset
 * Card"), but the tab id is the same for all purposes so the `?tab=` URL is
 * stable across fileset types.
 */
export enum FilesetDetailTab {
  Card = 'card',
  Files = 'files',
}

export const isFilesetDetailTab = (value: string | undefined): value is FilesetDetailTab =>
  Object.values(FilesetDetailTab).includes(value as FilesetDetailTab);
