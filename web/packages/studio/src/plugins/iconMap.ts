// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { iconNames, type IconName } from 'lucide-react/dynamic';

const PLUGIN_ICON_NAMES: ReadonlySet<string> = new Set(iconNames);

export const isPluginIconName = (iconName: string): iconName is IconName =>
  PLUGIN_ICON_NAMES.has(iconName);
