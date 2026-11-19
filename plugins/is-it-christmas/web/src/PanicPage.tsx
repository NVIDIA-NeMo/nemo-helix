// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { Link } from 'react-router';
import { pluginPath } from './paths';
import { useRustPanic } from './queries';
import type { PluginHost } from './types';

/** Shows the error a Rust handler panic produces, as the server reported it. */
export function PanicPage({ host }: { host: PluginHost }) {
  const { data, isPending, isError, error } = useRustPanic(host);

  return (
    <Flex align="center" justify="center" className="h-full p-6">
      <Stack gap="4" style={{ maxWidth: 880 }}>
        <Text kind="label/bold/lg">The Rust server panicked</Text>
        {isPending && (
          <Text kind="body/regular/md" color="secondary">
            Asking it to panic…
          </Text>
        )}
        {isError && (
          <Text kind="body/regular/sm" color="secondary">
            {error instanceof Error ? error.message : 'Unknown error'}. Is the Rust server running?
          </Text>
        )}
        {data && (
          <>
            <Stack gap="1" className="rounded border border-subtle bg-surface-sunken p-4" data-testid="panic">
              {data.detail.split('\n').map((line, i) => (
                <Text key={i} kind="body/regular/sm" className="font-mono" style={{ whiteSpace: 'pre-wrap' }}>
                  {line || '\u00a0'}
                </Text>
              ))}
            </Stack>
            <Text kind="body/regular/xs" color="secondary">
              HTTP 500 from the Rust server, through the Helix gateway. The panic was caught, so the server is still
              running. Rust printed the same text to its own stderr.
            </Text>
          </>
        )}
        <Link to={pluginPath(host.workspaceId)} className="text-subtle underline">
          ← Back
        </Link>
      </Stack>
    </Flex>
  );
}
