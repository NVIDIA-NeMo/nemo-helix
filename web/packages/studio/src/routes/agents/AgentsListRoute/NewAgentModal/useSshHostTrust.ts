// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ScanSshHostKeysResponse } from '@nemo/sdk/generated/platform/schema';
import { useSshHostKeys } from '@studio/api/files/useSshHostKeys';
import { sshHostOf } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/git';
import type { UseQueryResult } from '@tanstack/react-query';
import { useCallback, useState } from 'react';

export interface SshHostTrust {
  hostKeys: UseQueryResult<ScanSshHostKeysResponse, unknown>;
  /** The lookup answers for the host in the URL as it stands now. */
  settled: boolean;
  /** `host[:port]` of the URL, as the allowlist names it. */
  host: string | undefined;
  trusted: boolean;
  setTrusted: (trusted: boolean) => void;
  /** The confirmed known_hosts line, or `''` until the user trusts the key shown. */
  knownHosts: string;
  reset: () => void;
}

/** ssh's first-connection prompt as state: only the key shown, for the host in the URL now, is trusted. */
export const useSshHostTrust = (workspace: string, url: string | undefined): SshHostTrust => {
  const { scan: hostKeys, settled } = useSshHostKeys(workspace, url);
  const [trustedLine, setTrustedLine] = useState<string | null>(null);
  const shownLine = settled ? hostKeys.data?.keys[0]?.known_hosts_line : undefined;
  // Trust is bound to the exact key shown, so a new host or a changed key asks again.
  const trusted = Boolean(shownLine) && trustedLine === shownLine;
  const setTrusted = useCallback(
    (next: boolean) => setTrustedLine(next ? (shownLine ?? null) : null),
    [shownLine]
  );
  const reset = useCallback(() => setTrustedLine(null), []);
  return {
    hostKeys,
    settled,
    host: url ? sshHostOf(url) : undefined,
    trusted,
    setTrusted,
    knownHosts: trusted ? (shownLine ?? '') : '',
    reset,
  };
};
