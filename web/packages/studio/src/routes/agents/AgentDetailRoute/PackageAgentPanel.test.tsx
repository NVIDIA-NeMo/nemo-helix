// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { PackageAgentPanel } from '@studio/routes/agents/AgentDetailRoute/PackageAgentPanel';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const workspace = workspace1.workspace;
const agent = 'my-agent';
const jobsUrl = `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/jobs/package`;
const jobUrl = `${jobsUrl}/:name`;

const renderPanel = (props?: {
  canPackage?: boolean;
  onImageBuilt?: (image: string) => void;
  onImageAvailable?: (image: string) => void;
}) =>
  renderRoute(
    <PackageAgentPanel
      workspace={workspace}
      agentName={agent}
      canPackage={props?.canPackage ?? true}
      onImageBuilt={props?.onImageBuilt}
      onImageAvailable={props?.onImageAvailable}
    />
  );

/** A submitted job that reports *status*, with the tag behind the result artifact. */
const mockJob = (status: string, image = 'nemo-agents/default/my-agent:1.0') => {
  server.use(
    http.post(jobsUrl, () => HttpResponse.json({ name: 'pkg-1', status: 'created' })),
    http.get(`${jobUrl}/status`, () => HttpResponse.json({ status })),
    http.get(`${jobUrl}/logs`, () => HttpResponse.json({ data: [] })),
    http.get(`${jobUrl}/results/package_result/download`, () =>
      HttpResponse.json({ image, agent, published: '' })
    )
  );
};

const clickBuild = async () => {
  const user = userEvent.setup();
  await user.click(screen.getByRole('button', { name: 'Build image' }));
};

describe('PackageAgentPanel', () => {
  it('pushes to a registry when one is given', async () => {
    const submitted: { registry?: string }[] = [];
    server.use(
      http.post(jobsUrl, async ({ request }) => {
        const body = (await request.json()) as { spec?: { registry?: string } };
        submitted.push({ registry: body.spec?.registry });
        return HttpResponse.json({ name: 'pkg-1', status: 'created' });
      }),
      http.get(`${jobUrl}/status`, () => HttpResponse.json({ status: 'active' }))
    );
    const user = userEvent.setup();
    renderPanel();

    await user.type(screen.getByRole('textbox', { name: /Registry/ }), '  nvcr.io/my-org  ');
    await clickBuild();

    await waitFor(() => expect(submitted).toHaveLength(1));
    expect(submitted[0].registry).toBe('nvcr.io/my-org');
  });

  it('omits the registry when the field is left empty', async () => {
    const submitted: Record<string, unknown>[] = [];
    server.use(
      http.post(jobsUrl, async ({ request }) => {
        const body = (await request.json()) as { spec?: Record<string, unknown> };
        submitted.push(body.spec ?? {});
        return HttpResponse.json({ name: 'pkg-1', status: 'created' });
      }),
      http.get(`${jobUrl}/status`, () => HttpResponse.json({ status: 'active' }))
    );
    renderPanel();

    await clickBuild();

    await waitFor(() => expect(submitted).toHaveLength(1));
    expect(submitted[0]).not.toHaveProperty('registry');
  });

  it('offers the job rather than a wall of build output', async () => {
    mockJob('active');
    renderPanel();

    await clickBuild();

    expect(await screen.findByRole('button', { name: 'View job' })).toBeInTheDocument();
    expect(screen.getByLabelText('Building image')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /download/i })).not.toBeInTheDocument();
  });

  it('surfaces the built tag once the job completes', async () => {
    mockJob('completed');
    renderPanel();
    await clickBuild();

    expect(await screen.findByText('nemo-agents/default/my-agent:1.0')).toBeInTheDocument();
  });

  it('hands the tag to the deployment flow', async () => {
    const onImageBuilt = vi.fn();
    mockJob('completed');
    renderPanel({ onImageBuilt });

    await clickBuild();
    const user = userEvent.setup();
    await user.click(await screen.findByRole('button', { name: 'Use for deployment' }));

    expect(onImageBuilt).toHaveBeenCalledWith('nemo-agents/default/my-agent:1.0');
  });

  it('reports the tag without waiting to be asked, so deploying elsewhere has a default', async () => {
    const onImageAvailable = vi.fn();
    mockJob('completed');
    renderPanel({ onImageAvailable });

    await clickBuild();
    await screen.findByText('nemo-agents/default/my-agent:1.0');

    expect(onImageAvailable).toHaveBeenCalledWith('nemo-agents/default/my-agent:1.0');
  });

  it('reports no tag when the job finishes without one', async () => {
    const onImageAvailable = vi.fn();
    mockJob('completed', '');
    renderPanel({ onImageAvailable });

    await clickBuild();
    await screen.findByText(/finished without reporting an image tag/);

    expect(onImageAvailable).not.toHaveBeenCalled();
  });

  it('shows the rejection reason rather than a generic failure', async () => {
    server.use(
      http.post(jobsUrl, () =>
        HttpResponse.json(
          {
            detail:
              "Agent 'my-agent' has config_format 'nat-workflow-v1'; platform-side packaging supports 'nemo-agents-spec-v1' only.",
          },
          { status: 422 }
        )
      )
    );
    renderPanel();

    await clickBuild();

    expect(
      await screen.findByText(/packaging supports 'nemo-agents-spec-v1' only/)
    ).toBeInTheDocument();
  });

  it('reports a failed build instead of waiting forever', async () => {
    mockJob('error');
    renderPanel();

    await clickBuild();

    expect(await screen.findByText(/Packaging failed/)).toBeInTheDocument();
  });

  it('does not offer a tag when the result carries none', async () => {
    server.use(
      http.post(jobsUrl, () => HttpResponse.json({ name: 'pkg-1', status: 'created' })),
      http.get(`${jobUrl}/status`, () => HttpResponse.json({ status: 'completed' })),
      http.get(`${jobUrl}/logs`, () => HttpResponse.json({ data: [] })),
      http.get(`${jobUrl}/results/package_result/download`, () => HttpResponse.json({ agent }))
    );
    const onImageBuilt = vi.fn();
    renderPanel({ onImageBuilt });

    await clickBuild();

    expect(await screen.findByText(/finished without reporting an image tag/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Use for deployment' })).not.toBeInTheDocument();
  });

  it('says queued while the job waits to be dispatched', async () => {
    mockJob('created');
    renderPanel();

    await clickBuild();

    expect(await screen.findByRole('button', { name: 'Queued…' })).toBeInTheDocument();
  });

  it('refuses to package a NAT workflow agent', async () => {
    renderPanel({ canPackage: false });

    expect(screen.getByRole('button', { name: 'Build image' })).toBeDisabled();
    expect(
      await screen.findByText(/Packaging is available for Platform-managed agents/)
    ).toBeInTheDocument();
  });
});
