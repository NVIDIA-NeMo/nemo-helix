// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Flex,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@nvidia/foundations-react-core';
import type { FC, PropsWithChildren, ReactNode } from 'react';

export interface AccordionSectionProps {
  icon?: ReactNode;
  isDisabled?: boolean;
  title: string;
  value: string;
  className?: string;
  contentClassName?: string;
}

/**
 * AccordionItem implementation to support a custom icon in the header of an accordion section.
 */
export const AccordionSection: FC<PropsWithChildren<AccordionSectionProps>> = ({
  children,
  icon,
  isDisabled = false,
  title,
  value,
  className,
  contentClassName,
}) => {
  return (
    <AccordionItem value={value} disabled={isDisabled} className={className}>
      <AccordionTrigger>
        <Flex align="center" gap="2">
          {icon}
          {title}
        </Flex>
      </AccordionTrigger>
      <AccordionContent className={contentClassName}>{children}</AccordionContent>
    </AccordionItem>
  );
};
