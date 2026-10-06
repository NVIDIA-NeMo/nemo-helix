// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage, isBadRequestError } from '@nemo/common/src/api/common/utils';
import type { ScanSshHostKeysResponse } from '@nemo/sdk/generated/platform/schema';
import { Banner, Button, Checkbox, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { sshKeyTypeLabel } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/git';
import type { UseQueryResult } from '@tanstack/react-query';
import { RefreshCw } from 'lucide-react';
import type { FC } from 'react';

interface HostKeyConfirmationProps {
  hostKeys: UseQueryResult<ScanSshHostKeysResponse, unknown>;
  /** False while the lookup still answers for an earlier host, or has not answered yet. */
  settled: boolean;
  trusted: boolean;
  onTrustedChange: (trusted: boolean) => void;
  /** `host[:port]` from the URL, named when the host is refused. */
  host: string | undefined;
  disabled: boolean;
}

/** ssh's first-connection prompt: show the fingerprint, and trust the host only once confirmed. */
export const HostKeyConfirmation: FC<HostKeyConfirmationProps> = ({
  hostKeys,
  settled,
  trusted,
  onTrustedChange,
  host,
  disabled,
}) => {
  if (!settled) {
    return (
      <Text kind="body/regular/sm" color="subtle">
        Fetching host key…
      </Text>
    );
  }

  if (hostKeys.isError) {
    const message = getErrorMessage(hostKeys.error as Error);
    if (isBadRequestError(hostKeys.error)) {
      // Retrying cannot help a refused host until an administrator changes the configuration.
      return (
        <Text kind="body/regular/sm" color="subtle">
          {message.includes('allowed list') && host ? (
            <>
              {host} isn&apos;t an allowed host. Ask an administrator to add{' '}
              <span className="font-mono">ssh://{host}</span> to the files service&apos;s allowed
              external hosts.
            </>
          ) : (
            message
          )}
        </Text>
      );
    }
    return (
      <Flex align="center" gap="density-sm">
        <Text kind="body/regular/sm" color="subtle">
          Could not look up the host key: {message}
        </Text>
        <Button
          kind="secondary"
          size="small"
          type="button"
          disabled={disabled}
          onClick={() => void hostKeys.refetch()}
        >
          <RefreshCw size={14} aria-hidden />
          Retry
        </Button>
      </Flex>
    );
  }

  const primary = hostKeys.data?.keys[0];
  if (!hostKeys.data || !primary) return null;

  return (
    <Stack
      gap="density-md"
      className="rounded-lg bg-surface-sunken p-density-lg"
      data-testid="host-key-confirmation"
    >
      <Banner kind="inline" status="warning">
        The authenticity of host &apos;{hostKeys.data.host}&apos; cannot be established.
      </Banner>
      <Stack gap="density-xs">
        <Text kind="label/bold/sm" color="secondary">
          {sshKeyTypeLabel(primary.key_type)} key fingerprint
        </Text>
        <Text kind="mono/md" className="break-all">
          {primary.fingerprint}
        </Text>
      </Stack>
      <Checkbox
        checked={trusted}
        disabled={disabled}
        onCheckedChange={(checked) => onTrustedChange(checked === true)}
        slotLabel="I trust this host"
      />
    </Stack>
  );
};
