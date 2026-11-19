// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { useIsItChristmas } from './queries';
import type { PluginHost } from './types';

export function IsItChristmasPage({ host }: { host: PluginHost }) {
  const { data, isPending, isError, error } = useIsItChristmas(host);

  return (
    <Flex align="center" justify="center" className="h-full p-6">
      <Stack gap="3" align="center" data-testid="answer">
        {isPending && <Text kind="body/regular/md" color="secondary">Checking the calendar…</Text>}
        {isError && (
          <>
            <Text kind="label/bold/lg">Couldn't reach the plugin backend</Text>
            <Text kind="body/regular/sm" color="secondary">
              {error instanceof Error ? error.message : 'Unknown error'}. Is the Rust server running?
            </Text>
          </>
        )}
        {data && (
          <>
            <Text kind="label/bold/lg">{data.is_christmas ? 'YES!' : 'NO'}</Text>
            <Text kind="body/regular/md">{data.message}</Text>
            <Text kind="body/regular/xs" color="secondary">
              {data.date} · next Christmas {data.next_christmas}
            </Text>
          </>
        )}
        <Text kind="body/regular/xs" color="secondary">
          Answered by a Rust server on the other side of the Helix gateway.
        </Text>
      </Stack>
    </Flex>
  );
}
