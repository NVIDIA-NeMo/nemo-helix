// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { Anchor, CodeSnippet, Stack, StatusMessage, Text } from '@nvidia/foundations-react-core';
import { GlobeX } from 'lucide-react';
import type { FC } from 'react';

interface HelixUnavailableProps {
  /** URL Studio probed, shown so the user can confirm it points at the right platform. */
  healthUrl: string;
  onRetry: () => void;
  isRetrying?: boolean;
}

export const HelixUnavailable: FC<HelixUnavailableProps> = ({
  healthUrl,
  onRetry,
  isRetrying = false,
}) => {
  return (
    <Stack gap="density-xl" align="center" justify="center" className="h-screen px-density-lg">
      <StatusMessage
        className="max-w-[640px]"
        slotMedia={<GlobeX className="size-16 stroke-2" />}
        slotHeading="Studio can't connect to NeMo Helix"
        slotSubheading={
          <>
            Studio could not confirm a connection to the platform.
            <br />
            Make sure the platform is running and that Studio is pointed at it. Studio will retry
            automatically.
          </>
        }
        slotFooter={
          <LoadingButton color="brand" onClick={onRetry} loading={isRetrying} disabled={isRetrying}>
            Retry connection
          </LoadingButton>
        }
      />
      <Stack
        gap="density-sm"
        align="center"
        className="w-full max-w-[640px] text-center"
        data-testid="helix-unavailable-help"
      >
        <CodeSnippet
          value="nemo services run"
          language="bash"
          kind="block"
          className="w-full text-left"
          slotActions={
            <Text kind="body/regular/sm" color="secondary" className="mr-auto">
              Running locally? Start the platform, then retry:
            </Text>
          }
        />
        <Text kind="body/regular/sm" color="secondary">
          Studio checked{' '}
          <Anchor href={healthUrl} kind="inline" target="_blank" rel="noopener">
            {healthUrl}
          </Anchor>
        </Text>
      </Stack>
    </Stack>
  );
};
