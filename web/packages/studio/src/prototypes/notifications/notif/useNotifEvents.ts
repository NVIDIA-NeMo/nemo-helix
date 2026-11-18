// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { logger } from '@nemo/common/src/utils/logger';
import { withHelixBrowserAuthRequest } from '@nemo/sdk/src/utils/platformRequest';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import type { NotifEvent } from '@studio/prototypes/notifications/notif/types';
import { notifStreamPath, parseNotifEvent } from '@studio/prototypes/notifications/notif/utils';
import { useOidcBearerToken } from '@studio/providers/auth/useOidcBearerToken';
import { streamSse } from '@studio/util/sseStream';
import { useEffect, useRef } from 'react';

interface UseNotifEventsOptions {
  enabled?: boolean;
}

export const useNotifEvents = (
  workspace: string | undefined,
  onEvent: (event: NotifEvent) => void,
  { enabled = true }: UseNotifEventsOptions = {}
): void => {
  const accessToken = useOidcBearerToken();
  const onEventRef = useRef(onEvent);

  useEffect(() => {
    onEventRef.current = onEvent;
  }, [onEvent]);

  useEffect(() => {
    if (!workspace || !enabled) {
      return;
    }
    const url = `${PLATFORM_BASE_URL}${notifStreamPath(workspace)}`;
    const controller = new AbortController();

    void (async () => {
      try {
        const request = await withHelixBrowserAuthRequest({
          headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : undefined,
        });
        if (controller.signal.aborted) {
          return;
        }
        await streamSse(url, {
          signal: controller.signal,
          credentials: request.credentials,
          headers: request.headers,
          onEvent: ({ data }) => {
            const event = parseNotifEvent(data);
            if (event) {
              onEventRef.current(event);
            }
          },
          onError: (error) => {
            logger.warn(`Notification stream for workspace ${workspace} interrupted`, error);
          },
        });
      } catch (error) {
        if (!controller.signal.aborted) {
          logger.warn(`Unable to start notification stream for workspace ${workspace}`, error);
        }
      }
    })();

    return () => controller.abort();
  }, [workspace, enabled, accessToken]);
};
