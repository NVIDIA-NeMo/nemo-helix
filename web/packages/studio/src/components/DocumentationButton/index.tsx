// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button } from '@nvidia/foundations-react-core';
import type { ComponentProps, FC } from 'react';

type DocumentationButtonProps = {
  href: string;
  text?: string;
  attributes?: {
    Button?: ComponentProps<typeof Button>;
  };
};

/**
 * Tertiary KUI button for linking out to external documentation.
 */
export const DocumentationButton: FC<DocumentationButtonProps> = ({
  href,
  text = 'Documentation',
  attributes = {},
}) => {
  return (
    <Button kind="tertiary" asChild {...attributes.Button}>
      <a href={href} target="_blank" rel="noopener noreferrer">
        {text}
      </a>
    </Button>
  );
};
