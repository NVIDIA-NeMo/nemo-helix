// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createContext, useContext } from 'react';

export type MessageTextSize = 'md' | 'lg';

const MESSAGE_TEXT_STYLES = {
  md: { kind: 'body/regular/md', className: 'text-sm leading-6' },
  // KUI maps `text-base` to a text color token, so the 16px size uses `text-16`.
  lg: { kind: 'body/regular/lg', className: 'text-16 leading-6' },
} as const;

export type MessageTextStyle = (typeof MESSAGE_TEXT_STYLES)[MessageTextSize];

export const DEFAULT_MESSAGE_TEXT_STYLE: MessageTextStyle = MESSAGE_TEXT_STYLES.md;

export const MessageTextSizeContext = createContext<MessageTextSize>('md');

export const useMessageTextStyle = (): MessageTextStyle =>
  MESSAGE_TEXT_STYLES[useContext(MessageTextSizeContext)];
