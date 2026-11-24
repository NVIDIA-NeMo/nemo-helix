// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react';

/** How long Felix holds up his notepad after the assistant asks the user something. */
export const NOTEPAD_MS = 10_000;

/**
 * Tells Felix when to pull out his notepad: for a few seconds after the assistant starts waiting
 * on the user (a permission or input request).
 *
 * - `requestKey` identifies the pending request (null when there is none). A different request
 *   starts a fresh spell, so a second question in a row gets its own notepad.
 * - The request being answered or dismissed puts the notepad away straight away.
 * - After `NOTEPAD_MS` it goes back to normal even if the request is still open; the attention
 *   badge keeps saying he needs the user.
 */
export const useFelixNotepad = (requestKey: string | null): boolean => {
  const [isShowing, setIsShowing] = useState(false);

  useEffect(() => {
    if (requestKey === null) {
      setIsShowing(false);
      return;
    }

    setIsShowing(true);
    const timer = setTimeout(() => setIsShowing(false), NOTEPAD_MS);
    return () => clearTimeout(timer);
  }, [requestKey]);

  return isShowing;
};
