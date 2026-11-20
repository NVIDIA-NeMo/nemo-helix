// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { Link } from 'react-router';
import { pluginPath } from './paths';
import { useIsItChristmas, usePicture } from './queries';
import type { PluginHost } from './types';

export function IsItChristmasPage({ host }: { host: PluginHost }) {
  const { data, isPending, isError, error } = useIsItChristmas(host);
  // YES gets the waving Stefan, NO gets the sad one. Wait for the answer to know which.
  const mood = data ? (data.is_christmas ? 'happy' : 'sad') : undefined;
  const stefan = usePicture(host, 'stefan', mood, mood !== undefined);

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
            {data.forced && (
              <Text kind="body/regular/xs" color="secondary">
                Forced for the demo by CHRISTMAS_OVERRIDE on the server
              </Text>
            )}
          </>
        )}
        {stefan.data && (
          <img
            src={stefan.data.data_url}
            alt={
              mood === 'happy'
                ? 'Stefan, a cartoon figure in a Santa hat and sunglasses, waving'
                : 'Stefan, a cartoon figure in a Santa hat and sunglasses, crying'
            }
            width={220}
            height={220}
            style={{ borderRadius: 16 }}
          />
        )}
        <Text kind="body/regular/xs" color="secondary">
          Answered by a Rust server on the other side of the Helix gateway.
        </Text>
        <Link to={`${pluginPath(host.workspaceId)}/panic`} className="text-subtle underline">
          Trigger a Rust panic →
        </Link>
      </Stack>
    </Flex>
  );
}
