// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getNvidiaGuiIconSpriteSymbol, type IconName } from '@nemo/icons';

export const isPluginIconName = (iconName: string): iconName is IconName =>
  getNvidiaGuiIconSpriteSymbol(iconName as IconName) !== undefined;
