// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type ReactNode, createContext, useContext } from 'react';

interface InlinePickerSlotValue {
  /** Element rendered next to "Upload More Files" inside ``SimpleFilesTable``
   *  when the picker is used inline. ``null`` when no slot is provided (the
   *  default modal flow). */
  trailingButton: ReactNode | null;
}

const InlinePickerSlotContext = createContext<InlinePickerSlotValue>({ trailingButton: null });

export const InlinePickerSlotProvider = InlinePickerSlotContext.Provider;

export const useInlinePickerSlot = (): InlinePickerSlotValue => useContext(InlinePickerSlotContext);
