// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import type { AgentImage } from '@nemo/sdk/generated/agents/schema/AgentImage';
import { AGENT_IMAGES_URL } from '@studio/mocks/handlers/agentImages';
import { server } from '@studio/mocks/node';
import { DeleteDeploymentModal } from '@studio/routes/agents/AgentDetailRoute/components/DeleteDeploymentModal';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const IMAGE = 'nemo-agents/default/calc-abc:0.1';

const deployment = (image: string): AgentDeployment =>
  ({ name: 'calc-1', workspace: 'default', status: 'running', image }) as AgentDeployment;

const serveImages = (images: AgentImage[]) =>
  server.use(http.get(AGENT_IMAGES_URL, () => HttpResponse.json({ data: images })));

const openModal = (image: string) => {
  const onDelete = vi.fn().mockResolvedValue(true);
  renderRoute(
    <DeleteDeploymentModal
      workspace="default"
      deployment={deployment(image)}
      onDelete={onDelete}
      onClose={vi.fn()}
    />
  );
  return onDelete;
};

describe('DeleteDeploymentModal', () => {
  it('removes an image nothing else uses by default', async () => {
    serveImages([{ id: 'sha256:a', tags: [IMAGE], in_use: true, deployments: ['calc-1'] }]);
    const onDelete = openModal(IMAGE);

    expect(await screen.findByRole('checkbox', { name: 'Also remove the image' })).toBeChecked();
    await userEvent.click(screen.getByRole('button', { name: 'Delete' }));

    await waitFor(() => expect(onDelete).toHaveBeenCalledWith(true));
  });

  it('keeps an image other deployments use by default', async () => {
    serveImages([
      { id: 'sha256:a', tags: [IMAGE], in_use: true, deployments: ['calc-1', 'calc-2'] },
    ]);
    const onDelete = openModal(IMAGE);

    expect(await screen.findByText(/Also used by calc-2/)).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'Also remove the image' })).not.toBeChecked();
    await userEvent.click(screen.getByRole('button', { name: 'Delete' }));

    await waitFor(() => expect(onDelete).toHaveBeenCalledWith(false));
  });

  it('lets the user opt out', async () => {
    serveImages([]);
    const onDelete = openModal(IMAGE);

    await userEvent.click(await screen.findByRole('checkbox', { name: 'Also remove the image' }));
    await userEvent.click(screen.getByRole('button', { name: 'Delete' }));

    await waitFor(() => expect(onDelete).toHaveBeenCalledWith(false));
  });

  it('never offers to remove an image the platform did not build', async () => {
    const onDelete = openModal('my-registry/agent:1');

    await userEvent.click(await screen.findByRole('button', { name: 'Delete' }));

    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
    await waitFor(() => expect(onDelete).toHaveBeenCalledWith(false));
  });
});
