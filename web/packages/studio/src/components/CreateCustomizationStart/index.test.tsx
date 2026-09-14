// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { CreateCustomizationStart } from '@studio/components/CreateCustomizationStart';
import { CUSTOMIZATION_TEMPLATES } from '@studio/constants/customizationTemplates';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { mockUseNavigate, mockUseParams } from '@studio/tests/util/mockUseParams';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import type { Mock } from 'vitest';

const HF_ROWS_URL = 'https://datasets-server.huggingface.co/rows';

/** A BIRD-SQL row — the shape every shipped recipe's converter reads. */
const HF_ROW = {
  schema: 'CREATE TABLE t (id INT);',
  question: 'q',
  evidence: 'hint',
  SQL: 'SELECT 1',
};

const hfRowsHandler = http.get(HF_ROWS_URL, ({ request }) => {
  const length = Number(new URL(request.url).searchParams.get('length') ?? '0');
  return HttpResponse.json({ rows: Array.from({ length }, () => ({ row: HF_ROW })) });
});

const renderStart = (onContinue: Mock = vi.fn()) => {
  render(
    <TestProviders>
      <CreateCustomizationStart workspace={DEFAULT_WORKSPACE} onContinue={onContinue} />
    </TestProviders>
  );
};

const continueButton = () => screen.getByRole('button', { name: /continue/i });

describe('CreateCustomizationStart', () => {
  beforeEach(() => {
    mockUseNavigate(vi.fn());
    mockUseParams({ [ROUTE_PARAMS.workspace]: DEFAULT_WORKSPACE });
    server.use(hfRowsHandler);
  });

  it('offers a way in for each start option', () => {
    renderStart();
    expect(screen.getByText('Start from a template')).toBeInTheDocument();
    expect(screen.getByText('Build from scratch')).toBeInTheDocument();
  });

  it('keeps Continue disabled until an option is picked', async () => {
    const user = userEvent.setup();
    renderStart();
    expect(screen.queryByRole('button', { name: /continue/i })).not.toBeInTheDocument();

    await user.click(screen.getByText('Build from scratch'));
    expect(continueButton()).toBeEnabled();
  });

  it('hands "from scratch" over without any form values', async () => {
    const user = userEvent.setup();
    const onContinue = vi.fn();
    renderStart(onContinue);

    await user.click(screen.getByText('Build from scratch'));
    await user.click(continueButton());

    expect(onContinue).toHaveBeenCalledWith({ optionId: 'scratch' });
  });

  describe('templates', () => {
    it('shows a card per recipe once the option is picked', async () => {
      const user = userEvent.setup();
      renderStart();

      await user.click(screen.getByText('Start from a template'));
      for (const template of CUSTOMIZATION_TEMPLATES) {
        expect(screen.getByText(template.title)).toBeInTheDocument();
      }
    });

    /** Picking the option is not picking a recipe — Continue has nothing to act on yet. */
    it('will not continue until a recipe is selected', async () => {
      const user = userEvent.setup();
      renderStart();

      await user.click(screen.getByText('Start from a template'));
      expect(continueButton()).toBeDisabled();

      await user.click(screen.getByText(CUSTOMIZATION_TEMPLATES[0].title));
      expect(continueButton()).toBeEnabled();
    });

    it('provisions the recipe and hands over the form values it produced', async () => {
      const user = userEvent.setup();
      const onContinue = vi.fn();
      renderStart(onContinue);

      await user.click(screen.getByText('Start from a template'));
      await user.click(screen.getByText(CUSTOMIZATION_TEMPLATES[0].title));
      await user.click(continueButton());

      await waitFor(
        () =>
          expect(onContinue).toHaveBeenCalledWith(
            expect.objectContaining({
              optionId: 'template',
              initialValues: expect.objectContaining({ backend: 'automodel' }),
            })
          ),
        { timeout: 10_000 }
      );
    });

    it('reports a failed dataset fetch and does not continue', async () => {
      server.use(
        http.get(HF_ROWS_URL, () => HttpResponse.json({ error: 'boom' }, { status: 500 }))
      );
      const user = userEvent.setup();
      const onContinue = vi.fn();
      renderStart(onContinue);

      await user.click(screen.getByText('Start from a template'));
      await user.click(screen.getByText(CUSTOMIZATION_TEMPLATES[0].title));
      await user.click(continueButton());

      expect(
        await screen.findByText(/Failed to fetch dataset from Hugging Face/i)
      ).toBeInTheDocument();
      expect(onContinue).not.toHaveBeenCalled();
    });

    /**
     * Provisioning takes long enough that the cards stay on screen behind a disabled
     * Continue. Changing the selection mid-flight used to leave the finished setup handing
     * the form a recipe the user had moved off.
     *
     * The mocked fetch would otherwise resolve before a click could land, so the response is
     * held open to make the in-flight window real rather than a race.
     */
    it('ignores clicks on the option cards while setup is running', async () => {
      let releaseFetch: () => void = () => {};
      const held = new Promise<void>((resolve) => {
        releaseFetch = resolve;
      });
      server.use(
        http.get(HF_ROWS_URL, async ({ request }) => {
          await held;
          const length = Number(new URL(request.url).searchParams.get('length') ?? '0');
          return HttpResponse.json({ rows: Array.from({ length }, () => ({ row: HF_ROW })) });
        })
      );

      const user = userEvent.setup();
      const onContinue = vi.fn();
      renderStart(onContinue);

      await user.click(screen.getByText('Start from a template'));
      await user.click(screen.getByText(CUSTOMIZATION_TEMPLATES[0].title));
      await user.click(continueButton());

      // Setup is now parked on the held fetch. The cards are still mounted and clickable.
      await user.click(screen.getByText('Build from scratch'));
      expect(screen.getByText(CUSTOMIZATION_TEMPLATES[0].title)).toBeInTheDocument();

      releaseFetch();
      await waitFor(
        () =>
          expect(onContinue).toHaveBeenCalledWith(
            expect.objectContaining({ optionId: 'template' })
          ),
        { timeout: 10_000 }
      );
      expect(onContinue).not.toHaveBeenCalledWith({ optionId: 'scratch' });
    });
  });
});
