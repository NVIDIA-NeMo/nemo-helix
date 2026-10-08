// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getFilesDownloadFileQueryKey } from '@nemo/sdk/generated/platform/files';
import { readAgentEthos } from '@studio/api/agents/agentEthos';
import { agentEthosHandlers } from '@studio/mocks/handlers/insights';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { http, HttpResponse } from 'msw';

const ETHOS = '# Ethos\n\n## Role\n\nTriage security email.\n';

describe('readAgentEthos', () => {
  it('reads ETHOS.md from the agent ethos fileset', async () => {
    server.use(...agentEthosHandlers('triage', ETHOS));

    await expect(readAgentEthos('default', 'triage')).resolves.toBe(ETHOS);
  });

  it('returns the current file rather than an earlier read', async () => {
    server.use(...agentEthosHandlers('triage', ETHOS));
    await readAgentEthos('default', 'triage');

    server.use(...agentEthosHandlers('triage', '# Revised'));

    await expect(readAgentEthos('default', 'triage')).resolves.toBe('# Revised');
  });

  it('returns undefined when the agent has no ethos fileset', async () => {
    server.use(...agentEthosHandlers('triage', null));

    await expect(readAgentEthos('default', 'triage')).resolves.toBeUndefined();
  });

  it('returns undefined for a blank ETHOS.md', async () => {
    server.use(...agentEthosHandlers('triage', '  \n'));

    await expect(readAgentEthos('default', 'triage')).resolves.toBeUndefined();
  });

  it('returns undefined when the read fails', async () => {
    server.use(
      http.head(
        mockApiUrl(getFilesDownloadFileQueryKey, ':workspace', 'triage-ethos', 'ETHOS.md'),
        () => new HttpResponse(null, { status: 500 })
      )
    );

    await expect(readAgentEthos('default', 'triage')).resolves.toBeUndefined();
  });
});
