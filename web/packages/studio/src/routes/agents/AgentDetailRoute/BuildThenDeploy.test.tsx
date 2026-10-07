// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type BuildThenDeployRequest,
  buildThenDeployNavigation,
} from '@studio/api/agents/buildThenDeploy';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import {
  BuildThenDeploy,
  type PendingImageBuild,
} from '@studio/routes/agents/AgentDetailRoute/BuildThenDeploy';
import { getAgentDetailRoute, getAgentsListRoute } from '@studio/routes/utils';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import { http, HttpResponse } from 'msw';
import { type FC, useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router';

const workspace = workspace1.workspace;
const agent = 'my-agent';
const jobsUrl = `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/jobs/package`;
const jobUrl = `${jobsUrl}/:name`;
const deploymentsUrl = `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/deployments`;
const BUILT_IMAGE = 'nemo-agents/default/my-agent:2.0';

const Launcher: FC<{ request: BuildThenDeployRequest }> = ({ request }) => {
  const navigate = useNavigate();
  useEffect(() => {
    navigate(...buildThenDeployNavigation(workspace, agent, request));
  }, [navigate, request]);
  return null;
};

const AgentPage = () => {
  const [pending, setPending] = useState<PendingImageBuild | null>(null);
  return (
    <>
      <div>{useLocation().state ? 'request pending' : 'request consumed'}</div>
      {pending ? (
        <div>{`Waiting on ${pending.jobName ?? 'submit'} for ${pending.mode}`}</div>
      ) : null}
      <BuildThenDeploy workspace={workspace} agentName={agent} onPendingChange={setPending} />
    </>
  );
};

const renderFrom = (start: string, request: BuildThenDeployRequest) =>
  renderRoute(undefined, {
    history: start,
    routes: [
      { path: ROUTES.workspace.agentsList, element: <Launcher request={request} /> },
      { path: ROUTES.workspace.agentDetail, element: <AgentPage /> },
    ],
  });

interface JobScenario {
  restored?: { name: string; status: string; image: string };
  status: string;
}

const mockPackagingAndDeploys = ({ restored, status }: JobScenario) => {
  const submitted: unknown[] = [];
  const deployments: unknown[] = [];
  const statusByJob: Record<string, string> = { 'pkg-new': status };
  const imageByJob: Record<string, string> = { 'pkg-new': BUILT_IMAGE };
  if (restored) {
    statusByJob[restored.name] = restored.status;
    imageByJob[restored.name] = restored.image;
  }
  server.use(
    http.get(jobsUrl, () =>
      HttpResponse.json({
        data: restored
          ? [{ name: restored.name, created_at: '2026-01-02T03:04:05Z', spec: { agent } }]
          : [],
        total: restored ? 1 : 0,
      })
    ),
    http.post(jobsUrl, async ({ request }) => {
      submitted.push(await request.json());
      return HttpResponse.json({ name: 'pkg-new', status: 'created' });
    }),
    http.get(`${jobUrl}/status`, ({ params }) =>
      HttpResponse.json({ status: statusByJob[String(params['name'])] })
    ),
    http.get(`${jobUrl}/results/package_result/download`, ({ params }) =>
      HttpResponse.json({ image: imageByJob[String(params['name'])], agent, published: '' })
    ),
    http.post(deploymentsUrl, async ({ request }) => {
      const body = (await request.json()) as { agent?: string };
      deployments.push(body);
      return HttpResponse.json({ ...body, name: 'dep-1', workspace }, { status: 201 });
    })
  );
  return { submitted, deployments };
};

describe('BuildThenDeploy', () => {
  it('builds an image and deploys the tag with the requested runtime and name', async () => {
    const { submitted, deployments } = mockPackagingAndDeploys({ status: 'completed' });
    renderFrom(getAgentsListRoute(workspace), { mode: 'docker', deploymentName: 'prod' });

    await waitFor(() =>
      expect(deployments).toEqual([
        { agent, deployment_mode: 'docker', image: BUILT_IMAGE, name: 'prod' },
      ])
    );
    expect(submitted).toHaveLength(1);
    expect(screen.getByText('request consumed')).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText(/Waiting on/)).not.toBeInTheDocument());
  });

  it("never deploys an earlier build's image while its own build runs", async () => {
    const { submitted, deployments } = mockPackagingAndDeploys({
      restored: { name: 'pkg-old', status: 'completed', image: 'nemo-agents/default/my-agent:1.0' },
      status: 'running',
    });
    renderFrom(getAgentsListRoute(workspace), { mode: 'k8s' });

    await waitFor(() => expect(submitted).toHaveLength(1));
    expect(await screen.findByText(/Building an image for "my-agent"/)).toBeInTheDocument();
    expect(await screen.findByText('Waiting on pkg-new for k8s')).toBeInTheDocument();
    expect(deployments).toEqual([]);
  });

  it('reports a failed build instead of deploying', async () => {
    const { deployments } = mockPackagingAndDeploys({ status: 'error' });
    renderFrom(getAgentsListRoute(workspace), { mode: 'docker' });

    expect(await screen.findByText(/did not produce an image to deploy/)).toBeInTheDocument();
    expect(deployments).toEqual([]);
  });

  it('builds nothing on a plain visit', async () => {
    const { submitted } = mockPackagingAndDeploys({ status: 'completed' });
    renderRoute(undefined, {
      history: getAgentDetailRoute(workspace, agent),
      routes: [{ path: ROUTES.workspace.agentDetail, element: <AgentPage /> }],
    });

    expect(await screen.findByText('request consumed')).toBeInTheDocument();
    expect(submitted).toEqual([]);
  });
});
