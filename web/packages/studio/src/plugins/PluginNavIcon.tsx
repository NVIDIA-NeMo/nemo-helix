// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NvidiaGuiIcon } from '@nemo/icons';
import { isPluginIconName } from '@studio/plugins/iconMap';
import { Component, type FC, type ReactNode } from 'react';

interface PluginNavIconProps {
  readonly iconName: string;
  readonly className?: string;
}

interface ErrorBoundaryState {
  readonly hasError: boolean;
}

class PluginNavIconErrorBoundary extends Component<
  { readonly children: ReactNode },
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }

  override render() {
    return this.state.hasError ? null : this.props.children;
  }
}

const PluginNavIconInner: FC<PluginNavIconProps> = ({ iconName, className }) => {
  if (!isPluginIconName(iconName)) return null;
  return <NvidiaGuiIcon iconName={iconName} className={className} />;
};

export const PluginNavIcon: FC<PluginNavIconProps> = (props) => (
  <PluginNavIconErrorBoundary>
    <PluginNavIconInner {...props} />
  </PluginNavIconErrorBoundary>
);
