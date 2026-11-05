// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentImage } from '@nemo/sdk/generated/agents/schema/AgentImage';
import { AGENT_IMAGES_URL } from '@studio/mocks/handlers/agentImages';
import { server } from '@studio/mocks/node';
import { ImagesPanel } from '@studio/routes/agents/AgentDetailRoute/components/ImagesPanel';
import { renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { delay, http, HttpResponse } from 'msw';

const OLD = 'nemo-agents/default/calc-old:0.1';
const LIVE = 'nemo-agents/default/calc-live:0.2';

const images: AgentImage[] = [
  { id: 'sha256:old', tags: [OLD], in_use: false, size: 2048 },
  { id: 'sha256:live', tags: [LIVE], in_use: true, deployments: ['calc-1'] },
];

describe('ImagesPanel', () => {
  it('shows a loading state while fetching', async () => {
    server.use(
      http.get(AGENT_IMAGES_URL, async () => {
        await delay('infinite');
        return HttpResponse.json({ data: [] });
      })
    );
    renderRoute(<ImagesPanel workspace="default" agentName="calc" canPackage />);

    expect(await screen.findByTestId('images-loading')).toBeInTheDocument();
    expect(screen.getByText('Images')).toBeInTheDocument();
  });

  it('lists every image with its status and removes an unused one', async () => {
    const removed: string[] = [];
    const agents: (string | null)[] = [];
    server.use(
      http.get(AGENT_IMAGES_URL, ({ request }) => {
        agents.push(new URL(request.url).searchParams.get('agent'));
        return HttpResponse.json({ data: images.filter((i) => !removed.includes(i.id)) });
      }),
      http.delete(`${AGENT_IMAGES_URL}/:imageId`, ({ params }) => {
        removed.push(String(params.imageId));
        return new HttpResponse(null, { status: 204 });
      })
    );
    renderRoute(<ImagesPanel workspace="default" agentName="calc" canPackage />);

    expect(await screen.findByText(OLD)).toBeInTheDocument();
    expect(agents[0]).toBe('calc');
    const [oldRow, liveRow] = screen.getAllByTestId('image-row');
    expect(within(oldRow).getByText(OLD)).toBeInTheDocument();
    expect(within(oldRow).getByText('Unused')).toBeInTheDocument();
    expect(within(liveRow).getByText('In use')).toBeInTheDocument();
    expect(within(liveRow).getByText(/Used by calc-1/)).toBeInTheDocument();
    expect(within(liveRow).getByRole('button', { name: 'Remove' })).toBeDisabled();

    await userEvent.click(within(oldRow).getByRole('button', { name: 'Remove' }));
    await userEvent.click(screen.getByRole('button', { name: 'Delete' }));

    await waitFor(() => expect(removed).toEqual(['sha256:old']));
    await waitFor(() => expect(screen.queryByText(OLD)).not.toBeInTheDocument());
    expect(screen.getAllByTestId('image-row')).toHaveLength(1);
  });

  it('shows an empty state when the agent has no images', async () => {
    server.use(http.get(AGENT_IMAGES_URL, () => HttpResponse.json({ data: [] })));
    renderRoute(<ImagesPanel workspace="default" agentName="calc" canPackage />);

    expect(await screen.findByText('No images built for this agent.')).toBeInTheDocument();
  });
});
