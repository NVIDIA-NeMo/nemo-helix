// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isPluginIconName } from '@studio/plugins/iconMap';
import { DynamicIcon } from 'lucide-react/dynamic';
import type { FC } from 'react';

interface PluginNavIconProps {
  readonly iconName: string;
  readonly className?: string;
}

export const PluginNavIcon: FC<PluginNavIconProps> = ({ iconName, className }) =>
  isPluginIconName(iconName) ? <DynamicIcon name={iconName} className={className} /> : null;
