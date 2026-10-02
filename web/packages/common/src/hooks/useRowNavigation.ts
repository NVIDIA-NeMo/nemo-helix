// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from 'react';
import { useNavigate } from 'react-router';

const wantsNewTab = (event: React.MouseEvent): boolean =>
  event.metaKey || event.ctrlKey || event.button === 1;

export function useRowNavigation() {
  const navigate = useNavigate();

  return useCallback(
    (event: React.MouseEvent | undefined, href: string) => {
      if (event && wantsNewTab(event)) {
        window.open(href, '_blank', 'noopener,noreferrer');
        return;
      }
      navigate(href);
    },
    [navigate]
  );
}
