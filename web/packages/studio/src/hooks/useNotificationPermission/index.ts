// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from 'react';

export const isNotificationSupported = (): boolean => typeof Notification !== 'undefined';

export const isNotificationGranted = (): boolean =>
  isNotificationSupported() && Notification.permission === 'granted';

/**
 * Browser notification permission. The prompt is only ever raised from `requestIfNeeded`, and
 * only while the answer is still undecided: a granted or denied choice is never asked again.
 */
export const useNotificationPermission = () => {
  const requestIfNeeded = useCallback(async (): Promise<void> => {
    if (!isNotificationSupported() || Notification.permission !== 'default') return;
    try {
      await Notification.requestPermission();
    } catch {
      // Some browsers refuse a request made outside a user gesture; the user can still opt in
      // from the browser's site settings.
    }
  }, []);

  return { requestIfNeeded };
};
