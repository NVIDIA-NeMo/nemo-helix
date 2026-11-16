// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CALLOUT_CONFIG } from '@nemo/common/src/components/MarkdownContent/constants';
import { isCalloutKind } from '@nemo/common/src/components/MarkdownContent/utils';
import { Text } from '@nvidia/foundations-react-core';
import cn from 'classnames';
import type { ComponentProps, FC } from 'react';
import type { ExtraProps } from 'react-markdown';

type BlockquoteProps = ComponentProps<'blockquote'> & ExtraProps;

// Renders either a styled GitHub-style callout (when the upstream
// `remarkCallouts` plugin tagged the node with `data-callout="<kind>"`) or a
// plain blockquote.
export const Blockquote: FC<BlockquoteProps> = ({ children, className, node, ...props }) => {
  const kind = node?.properties?.dataCallout;

  if (!isCalloutKind(kind)) {
    return (
      <blockquote
        className={cn(
          'mb-density-md border-l-4 border-base pl-density-md text-secondary',
          className
        )}
        {...props}
      >
        {children}
      </blockquote>
    );
  }

  const config = CALLOUT_CONFIG[kind];

  return (
    <blockquote
      className={cn(
        'mb-density-lg rounded-md border border-base border-l-4 bg-surface-raised p-density-md',
        config.borderClassName,
        className
      )}
      {...props}
    >
      <Text asChild kind="label/bold/sm" className={config.labelClassName}>
        <div className="mb-density-xs">{config.label}</div>
      </Text>
      <div className="[&>*:last-child]:mb-0">{children}</div>
    </blockquote>
  );
};
