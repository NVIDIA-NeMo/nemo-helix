// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ToastProvider } from '@nemo/common/src/providers/toast/ToastProvider';
import type { SampleAgentStreamEvent } from '@nemo/sdk/generated/agents/schema';
import type { ModelEntitysPage } from '@nemo/sdk/generated/platform/schema';
import type { Meta, StoryObj } from '@storybook/react';
import { CreateSampleAgentModal } from '@studio/routes/WorkspaceDashboardHomeRoute/CreateSampleAgentModal';
import { delay, http, HttpResponse } from 'msw';
import { AuthContext, type AuthContextProps } from 'react-oidc-context';

const SAMPLE_AGENT_API = '/apis/agents/v2/sample-agent';
const MODELS_API = '/apis/models/v2/workspaces/:workspace/models';
const WORKSPACE = 'sample-1a2b3c4d';

const modelsPage: ModelEntitysPage = {
  data: [
    {
      name: 'nemotron-3-super-120b',
      workspace: 'default',
      model_providers: ['default/build-nvidia'],
      created_at: '2026-09-01T00:00:00Z',
      updated_at: '2026-09-01T00:00:00Z',
    },
  ],
  pagination: { page: 1, page_size: 25, current_page_size: 1, total_pages: 1, total_results: 1 },
} as ModelEntitysPage;

/** Auth off: no signed-in user. */
const signedOut = { user: null } as unknown as AuthContextProps;

const progress = (
  component: NonNullable<SampleAgentStreamEvent['component']>,
  status: SampleAgentStreamEvent['status'] = 'created'
): SampleAgentStreamEvent => ({ kind: 'progress', component, status, workspace: WORKSPACE });

const stream = (...frames: SampleAgentStreamEvent[]) =>
  http.post(SAMPLE_AGENT_API, async () => {
    await delay(300);
    return new HttpResponse(frames.map((frame) => JSON.stringify(frame)).join('\n') + '\n', {
      status: 201,
      headers: { 'Content-Type': 'application/x-ndjson' },
    });
  });

const failBeforeStream = (status: number, body: string, contentType = 'application/json') =>
  http.post(SAMPLE_AGENT_API, async () => {
    await delay(300);
    return new HttpResponse(body, { status, headers: { 'Content-Type': contentType } });
  });

const scenario = (description: string, handler: ReturnType<typeof http.post>): Story => ({
  parameters: {
    docs: { description: { story: description } },
    msw: { handlers: [http.get(MODELS_API, () => HttpResponse.json(modelsPage)), handler] },
  },
});

const meta = {
  component: CreateSampleAgentModal,
  title: 'Routes/WorkspaceDashboardHomeRoute/CreateSampleAgentModal',
  args: { open: true, workspace: 'default', onClose: () => undefined },
  decorators: [
    (Story) => (
      <AuthContext.Provider value={signedOut}>
        <ToastProvider>
          <Story />
        </ToastProvider>
      </AuthContext.Provider>
    ),
  ],
} satisfies Meta<typeof CreateSampleAgentModal>;

export default meta;
type Story = StoryObj<typeof meta>;

// --- Rejected before the stream opens: a real HTTP error status --------------------------------

export const InvalidModelReference = scenario(
  '422: the model ref is not `workspace/name`. The picker always sends a qualified ref, so this is a guard.',
  failBeforeStream(
    422,
    JSON.stringify({ detail: "Model reference must be in format 'workspace/name'" })
  )
);

export const ModelNotFound = scenario(
  '404: the chosen model was deleted between picking it and submitting.',
  failBeforeStream(
    404,
    JSON.stringify({ detail: "Model 'default/nemotron-3-super-120b' not found" })
  )
);

export const Forbidden = scenario(
  '403: the caller may not create workspaces (auth on).',
  failBeforeStream(403, JSON.stringify({ detail: 'Forbidden' }))
);

export const PackagedAssetsUnavailable = scenario(
  '500: the agents plugin could not read the packaged sample agent assets.',
  failBeforeStream(500, JSON.stringify({ detail: 'Packaged sample assets are unavailable' }))
);

export const UnhandledServerError = scenario(
  '500: an unhandled exception before streaming (e.g. listing workspaces failed); FastAPI answers in plain text.',
  failBeforeStream(500, 'Internal Server Error', 'text/plain')
);

export const BadGatewayFromProxy = scenario(
  '502: an ingress proxy answers with its own HTML error page because the platform is down.',
  failBeforeStream(
    502,
    '<html>\r\n<head><title>502 Bad Gateway</title></head>\r\n<body>\r\n<center><h1>502 Bad Gateway</h1></center>\r\n<hr><center>nginx</center>\r\n</body>\r\n</html>\r\n',
    'text/html'
  )
);

export const ServiceUnavailableEmptyBody = scenario(
  '503: the service is unavailable and the response has no body.',
  failBeforeStream(503, '', 'text/plain')
);

export const GatewayTimeout = scenario(
  '504: the proxy timed out waiting for the platform.',
  failBeforeStream(504, 'upstream request timeout', 'text/plain')
);

export const NetworkError = scenario(
  'The request never got a response (offline, CORS, DNS).',
  http.post(SAMPLE_AGENT_API, () => HttpResponse.error())
);

// --- Failed after the stream opened: a 2xx response carrying an `error` frame ---------------------

export const AlreadyDeployedWithAnotherModel = scenario(
  'In-stream 409 at the agent step: the existing sample agent is deployed with a different model.',
  stream(progress('workspace', 'existing'), {
    kind: 'error',
    workspace: WORKSPACE,
    message: 'The sample agent has already been deployed with another model',
    status_code: 409,
  })
);

export const FailedAtAgentDeployment = scenario(
  'In-stream 502 at the agent step: creating the agent or its deployment failed upstream.',
  stream(progress('workspace'), {
    kind: 'error',
    workspace: WORKSPACE,
    message: 'Sample agent setup failed at agent deployment',
    status_code: 502,
    failed_step: 'agent deployment',
    retryable: true,
  })
);

export const FailedAtSampleFiles = scenario(
  'In-stream 502 at the files step: the agent is deployed but the dataset or eval config upload failed.',
  stream(progress('workspace'), progress('agent'), progress('deployment', 'submitted'), {
    kind: 'error',
    workspace: WORKSPACE,
    message: 'Sample agent setup failed at sample files',
    status_code: 502,
    failed_step: 'sample files',
    retryable: true,
  })
);

export const StreamEndedWithoutResult = scenario(
  'The connection dropped mid-stream: progress arrived but neither `done` nor `error` did.',
  stream(progress('workspace'), progress('agent'))
);
