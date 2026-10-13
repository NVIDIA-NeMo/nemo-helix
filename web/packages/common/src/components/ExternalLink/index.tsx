// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Anchor } from '@nvidia/foundations-react-core';
import { type ComponentProps, type FC, type ReactNode } from 'react';

interface ExternalLinkProps {
  href: string;
  textKind?: ComponentProps<typeof Anchor>['textKind'];
  children: ReactNode;
}

export const ExternalLink: FC<ExternalLinkProps> = ({ href, textKind, children }) => (
  <Anchor
    href={href}
    target="_blank"
    rel="noreferrer noopener"
    textKind={textKind}
    underline
    className="text-brand"
  >
    {children}
  </Anchor>
);
