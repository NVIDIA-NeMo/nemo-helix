// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMessageTextStyle } from '@nemo/common/src/components/Chat/MessageContent/textSize';
import { Text } from '@nvidia/foundations-react-core';
import cn from 'classnames';
import { type FC, type PropsWithChildren } from 'react';

export const MarkdownParagraph: FC<PropsWithChildren> = ({ children }) => {
  const textStyle = useMessageTextStyle();
  return (
    <Text asChild kind={textStyle.kind}>
      <p className={cn('mb-density-md last:mb-0', textStyle.className)}>{children}</p>
    </Text>
  );
};
