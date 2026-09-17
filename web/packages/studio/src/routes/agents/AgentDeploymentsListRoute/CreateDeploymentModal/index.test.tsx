// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_AGENT_CONTAINER_DEPLOYMENTS_ENABLED', 'true');
});

import { getAgentsListDeploymentsQueryKey } from '@nemo/sdk/generated/agents/agent-deployments';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { CreateDeploymentModal } from '@studio/routes/agents/AgentDeploymentsListRoute/CreateDeploymentModal';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import { within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { useState, type FC } from 'react';

const workspace = workspace1.workspace;
const agent = 'nemo-studio-assistant';
const deploymentsUrl = `${PLATFORM_BASE_URL}${getAgentsListDeploymentsQueryKey(':workspace')[0]}`;

interface CapturedDeployment {
  agent?: string;
  name?: string;
  deployment_mode?: string;
  image?: string;
}

const renderModal = (initialImage?: string) =>
  renderRoute(
    <CreateDeploymentModal
      open
      onClose={vi.fn()}
      workspace={workspace}
      agent={agent}
      initialImage={initialImage}
    />
  );

const captureCreate = (): { body?: CapturedDeployment } => {
  const captured: { body?: CapturedDeployment } = {};
  server.use(
    http.post(deploymentsUrl, async ({ request }) => {
      captured.body = (await request.json()) as CapturedDeployment;
      return HttpResponse.json({
        ...captured.body,
        name: captured.body.name ?? 'generated-deployment',
        workspace,
        status: 'pending',
      });
    })
  );
  return captured;
};

const PACKAGED_IMAGE = 'nemo-agents/default/my-agent:1.0';

/** Drives `open` and `initialImage` the way the agent detail route does. */
const Harness: FC<{ startImage?: string }> = ({ startImage }) => {
  const [open, setOpen] = useState(true);
  const [image, setImage] = useState(startImage);
  return (
    <>
      <button type="button" onClick={() => setImage(PACKAGED_IMAGE)}>
        harness: build finished
      </button>
      <button type="button" onClick={() => setOpen(false)}>
        harness: close
      </button>
      <button type="button" onClick={() => setOpen(true)}>
        harness: open
      </button>
      <CreateDeploymentModal
        open={open}
        onClose={() => setOpen(false)}
        workspace={workspace}
        agent={agent}
        initialImage={image}
      />
    </>
  );
};

const getDeploymentDialog = async (): Promise<HTMLDialogElement> => {
  const dialog = await screen.findByRole('dialog', { name: 'Deploy Agent' });
  if (!(dialog instanceof HTMLDialogElement)) {
    throw new Error('Deploy Agent did not render as a dialog');
  }
  return dialog;
};

describe('CreateDeploymentModal', () => {
  it('creates a deployment without a name when the optional field is empty', async () => {
    const user = userEvent.setup();
    const captured = captureCreate();
    renderModal();

    const dialog = await getDeploymentDialog();
    await user.click(within(dialog).getByRole('button', { name: 'Deploy' }));

    await waitFor(() =>
      expect(captured.body).toEqual({
        agent,
        deployment_mode: 'subprocess',
      })
    );
  });

  it.each([
    ['deployment with spaces', 'Deployment name cannot contain spaces'],
    ['Invalid-deployment', 'Deployment name must be lowercase'],
  ])('does not submit the invalid deployment name %s', async (name, expectedError) => {
    const user = userEvent.setup();
    const captured = captureCreate();
    renderModal();

    const dialog = await getDeploymentDialog();
    await user.type(
      within(dialog).getByRole('textbox', { name: 'Deployment Name (optional)' }),
      name
    );
    await user.click(within(dialog).getByRole('button', { name: 'Deploy' }));

    expect(await within(dialog).findByText(new RegExp(expectedError))).toBeInTheDocument();
    expect(captured.body).toBeUndefined();
  });

  it('creates a Docker deployment with the selected container image', async () => {
    const user = userEvent.setup();
    const captured = captureCreate();
    renderModal();

    const dialog = await getDeploymentDialog();
    await user.click(within(dialog).getByRole('combobox', { name: 'Runtime' }));
    await user.click(await screen.findByRole('option', { name: 'Docker' }));
    await user.type(
      within(dialog).getByRole('textbox', { name: 'Container Image' }),
      'nvcr.io/example/nemo-studio-assistant:poc'
    );
    await user.click(within(dialog).getByRole('button', { name: 'Deploy' }));

    await waitFor(() =>
      expect(captured.body).toEqual({
        agent,
        deployment_mode: 'docker',
        image: 'nvcr.io/example/nemo-studio-assistant:poc',
      })
    );
  });

  it('deploys a freshly packaged image without retyping it', async () => {
    const user = userEvent.setup();
    const captured = captureCreate();
    renderModal('nemo-agents/default/nemo-studio-assistant:1.0');

    const dialog = await getDeploymentDialog();
    await user.click(within(dialog).getByRole('button', { name: 'Deploy' }));

    await waitFor(() =>
      expect(captured.body).toEqual({
        agent,
        deployment_mode: 'docker',
        image: 'nemo-agents/default/nemo-studio-assistant:1.0',
      })
    );
  });

  it('keeps container options out of the way until asked for', async () => {
    const user = userEvent.setup();
    renderModal();

    const dialog = await getDeploymentDialog();

    // The accordion hides its content rather than unmounting it, so this is about
    // what the user can see, not what React rendered.
    expect(within(dialog).getByRole('combobox', { name: 'Runtime' })).not.toBeVisible();

    await user.click(within(dialog).getByText(/Show Advanced/));

    expect(within(dialog).getByRole('combobox', { name: 'Runtime' })).toBeVisible();
  });

  it('opens the disclosure when a packaged image arrives, so the tag is visible', async () => {
    renderModal('nemo-agents/default/my-agent:1.0');

    const dialog = await getDeploymentDialog();

    expect(await within(dialog).findByRole('textbox', { name: 'Container Image' })).toHaveValue(
      'nemo-agents/default/my-agent:1.0'
    );
  });

  it('keeps what the user typed when a build finishes while the dialog is open', async () => {
    const user = userEvent.setup();
    renderRoute(<Harness />);

    const dialog = await getDeploymentDialog();
    await user.type(within(dialog).getByRole('textbox', { name: /Deployment Name/ }), 'my-run');
    await user.click(screen.getByRole('button', { name: 'harness: build finished' }));

    expect(within(dialog).getByRole('textbox', { name: /Deployment Name/ })).toHaveValue('my-run');
  });

  it('seeds the packaged image again on a later open', async () => {
    const user = userEvent.setup();
    renderRoute(<Harness startImage={PACKAGED_IMAGE} />);

    const dialog = await getDeploymentDialog();
    await user.clear(within(dialog).getByRole('textbox', { name: 'Container Image' }));
    await user.click(screen.getByRole('button', { name: 'harness: close' }));
    await user.click(screen.getByRole('button', { name: 'harness: open' }));

    expect(
      await within(await getDeploymentDialog()).findByRole('textbox', { name: 'Container Image' })
    ).toHaveValue(PACKAGED_IMAGE);
  });

  it("surfaces the server's refusal when no image resolves", async () => {
    server.use(
      http.post(deploymentsUrl, () =>
        HttpResponse.json(
          {
            detail:
              "deployment_mode 'docker' requires a container image. Set 'image' on the request, or configure 'deployments.default_image'.",
          },
          { status: 400 }
        )
      )
    );
    const user = userEvent.setup();
    renderModal();

    const dialog = await getDeploymentDialog();
    await user.click(within(dialog).getByRole('combobox', { name: 'Runtime' }));
    await user.click(await screen.findByRole('option', { name: 'Docker' }));
    await user.click(within(dialog).getByRole('button', { name: 'Deploy' }));

    expect(await within(dialog).findByText(/requires a container image/)).toBeInTheDocument();
  });

  it('submits without an image so a configured default can apply', async () => {
    const captured = captureCreate();
    const user = userEvent.setup();
    renderModal();

    const dialog = await getDeploymentDialog();
    await user.click(within(dialog).getByRole('combobox', { name: 'Runtime' }));
    await user.click(await screen.findByRole('option', { name: 'Docker' }));
    await user.click(within(dialog).getByRole('button', { name: 'Deploy' }));

    await waitFor(() => expect(captured.body?.deployment_mode).toBe('docker'));
    expect(captured.body?.image).toBeUndefined();
  });
});
