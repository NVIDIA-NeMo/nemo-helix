// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { PackageAgentControl } from '@studio/routes/agents/AgentDetailRoute/PackageAgentControl';
import { renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const workspace = workspace1.workspace;
const agent = 'my-agent';
const jobsUrl = `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/jobs/package`;
const jobUrl = `${jobsUrl}/:name`;

const renderControl = (props?: {
  canPackage?: boolean;
  isAgentLoading?: boolean;
  onImageBuilt?: (image: string) => void;
  onImageAvailable?: (image: string) => void;
}) =>
  renderRoute(
    <PackageAgentControl
      workspace={workspace}
      agentName={agent}
      canPackage={props?.canPackage ?? true}
      isAgentLoading={props?.isAgentLoading}
      onImageBuilt={props?.onImageBuilt}
      onImageAvailable={props?.onImageAvailable}
    />
  );

/** A build from an earlier page load, found by the restore query rather than submitted here. */
const mockRestoredJob = (
  status: string,
  createdAt = '2026-01-02T03:04:05Z',
  image = 'nemo-agents/default/my-agent:1.0'
) => {
  server.use(
    http.get(jobsUrl, () =>
      HttpResponse.json({
        data: [{ name: 'pkg-1', created_at: createdAt, spec: { agent } }],
        total: 1,
      })
    ),
    http.get(`${jobUrl}/status`, () => HttpResponse.json({ status })),
    http.get(`${jobUrl}/logs`, () => HttpResponse.json({ data: [] })),
    http.get(`${jobUrl}/results/package_result/download`, () =>
      HttpResponse.json({ image, agent, published: '' })
    )
  );
};

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

/** Packaging lives behind a button in the Deployments header, not a panel. */
const openControl = async (props?: Parameters<typeof renderControl>[0]) => {
  renderControl(props);
  const user = userEvent.setup();
  await user.click(screen.getByRole('button', { name: /Build image|Manage image/ }));
  return user;
};

/** The trigger and the modal's submit share a label, so this scopes to the dialog. */
const clickBuild = async () => {
  const user = userEvent.setup();
  await user.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Build image' }));
};

/** Build inputs sit behind a disclosure, so reaching the registry needs a click. */
const openPushOptions = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByText('Push options'));
  return screen.getByRole('textbox', { name: /Registry/ });
};

describe('PackageAgentControl', () => {
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
    const user = await openControl();

    await user.type(await openPushOptions(user), '  nvcr.io/my-org  ');
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
    await openControl();

    await clickBuild();

    await waitFor(() => expect(submitted).toHaveLength(1));
    expect(submitted[0]).not.toHaveProperty('registry');
  });

  it('offers the job rather than a wall of build output', async () => {
    mockJob('active');
    await openControl();

    await clickBuild();

    expect(await screen.findByRole('button', { name: 'View job' })).toBeInTheDocument();
    expect(screen.getByLabelText('Building image')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /download/i })).not.toBeInTheDocument();
  });

  it('surfaces the built tag once the job completes', async () => {
    mockJob('completed');
    await openControl();
    await clickBuild();

    expect(await screen.findByText('nemo-agents/default/my-agent:1.0')).toBeInTheDocument();
  });

  it('keeps the registry visible once an image exists, since a rebuild still pushes there', async () => {
    mockJob('completed');
    const user = await openControl();

    await user.type(await openPushOptions(user), 'nvcr.io/my-org');
    await clickBuild();
    await screen.findByText('nemo-agents/default/my-agent:1.0');

    expect(screen.getByText('Push options')).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: /Registry/ })).toHaveValue('nvcr.io/my-org');
  });

  it('hands the tag to the deployment flow', async () => {
    const onImageBuilt = vi.fn();
    mockJob('completed');
    await openControl({ onImageBuilt });

    await clickBuild();
    const user = userEvent.setup();
    await user.click(await screen.findByRole('button', { name: 'Deploy' }));

    expect(onImageBuilt).toHaveBeenCalledWith('nemo-agents/default/my-agent:1.0');
  });

  it('reports the tag without waiting to be asked, so deploying elsewhere has a default', async () => {
    const onImageAvailable = vi.fn();
    mockJob('completed');
    await openControl({ onImageAvailable });

    await clickBuild();
    await screen.findByText('nemo-agents/default/my-agent:1.0');

    expect(onImageAvailable).toHaveBeenCalledWith('nemo-agents/default/my-agent:1.0');
  });

  it('reports no tag when the job finishes without one', async () => {
    const onImageAvailable = vi.fn();
    mockJob('completed', '');
    await openControl({ onImageAvailable });

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
    await openControl();

    await clickBuild();

    expect(
      await screen.findByText(/packaging supports 'nemo-agents-spec-v1' only/)
    ).toBeInTheDocument();
  });

  it('reports a failed build instead of waiting forever', async () => {
    mockJob('error');
    await openControl();

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
    await openControl({ onImageBuilt });

    await clickBuild();

    expect(await screen.findByText(/finished without reporting an image tag/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Deploy' })).not.toBeInTheDocument();
  });

  it('says queued while the job waits to be dispatched', async () => {
    mockJob('created');
    await openControl();

    await clickBuild();

    expect(await screen.findByText(/Waiting for a build to start/)).toBeInTheDocument();
  });

  it('reports a running build on the trigger, so closing the modal does not lose it', async () => {
    mockJob('active');
    const user = await openControl();

    await clickBuild();
    await screen.findByText(/Building — this takes a few minutes/);
    await user.click(screen.getByRole('button', { name: 'Close' }));

    expect(await screen.findByRole('button', { name: /Building…/ })).toBeInTheDocument();
  });

  it('refuses to package a NAT workflow agent', async () => {
    await openControl({ canPackage: false });

    expect(
      within(screen.getByRole('dialog')).getByRole('button', { name: 'Build image' })
    ).toBeDisabled();
    expect(
      await screen.findByText(/Packaging is available for Platform-managed agents/)
    ).toBeInTheDocument();
  });

  it('does not call an unloaded agent a NAT workflow', async () => {
    await openControl({ canPackage: false, isAgentLoading: true });

    expect(
      await screen.findByText(/Checking whether this agent can be packaged/)
    ).toBeInTheDocument();
    expect(
      screen.queryByText(/Packaging is available for Platform-managed agents/)
    ).not.toBeInTheDocument();
  });

  it('stops implying progress when the status poll fails, so the build can be retried', async () => {
    server.use(
      http.post(jobsUrl, () => HttpResponse.json({ name: 'pkg-1', status: 'created' })),
      http.get(`${jobUrl}/status`, () => new HttpResponse(null, { status: 500 }))
    );
    await openControl();

    await clickBuild();

    expect(await screen.findByText(/Lost track of this build/)).toBeInTheDocument();
    expect(screen.queryByLabelText('Building image')).not.toBeInTheDocument();
    expect(
      within(screen.getByRole('dialog')).getByRole('button', { name: 'Build image' })
    ).toBeEnabled();
  });

  it('separates an unreadable result from a job that reported no tag', async () => {
    server.use(
      http.post(jobsUrl, () => HttpResponse.json({ name: 'pkg-1', status: 'created' })),
      http.get(`${jobUrl}/status`, () => HttpResponse.json({ status: 'completed' })),
      http.get(`${jobUrl}/logs`, () => HttpResponse.json({ data: [] })),
      http.get(
        `${jobUrl}/results/package_result/download`,
        () => new HttpResponse(null, { status: 500 })
      )
    );
    await openControl();

    await clickBuild();

    expect(await screen.findByText(/its result could not be read/)).toBeInTheDocument();
    expect(screen.queryByText(/finished without reporting an image tag/)).not.toBeInTheDocument();
  });

  it('still warns about a stalled queue after a reload lost the submit time', async () => {
    mockRestoredJob('created', '2026-01-02T03:04:05Z');
    await openControl();

    expect(await screen.findByText(/accepted but has not started/)).toBeInTheDocument();
  });

  it('does not make a restored image the silent default for the next deployment', async () => {
    const onImageAvailable = vi.fn();
    mockRestoredJob('completed');
    await openControl({ onImageAvailable });

    await screen.findByText('nemo-agents/default/my-agent:1.0');

    expect(onImageAvailable).not.toHaveBeenCalled();
    expect(screen.getByText(/rebuild to pick up newer changes/)).toBeInTheDocument();
  });
});
