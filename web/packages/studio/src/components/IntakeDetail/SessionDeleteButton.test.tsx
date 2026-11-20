// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import {
  getGetSessionDeletionPermissionQueryKey,
  getGetSessionQueryKey,
} from '@nemo/sdk/generated/platform/sessions';
import { SessionDeleteButton } from '@studio/components/IntakeDetail/SessionDeleteButton';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { getIntakeTracesRoute } from '@studio/routes/utils';
import { renderRoute, screen } from '@studio/tests/util/render';
import { waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const SESSION = 'session-a';
const PERMISSION = mockApiUrl(getGetSessionDeletionPermissionQueryKey, ':workspace', ':id');
const DELETE = mockApiUrl(getGetSessionQueryKey, ':workspace', ':id');

const show = () =>
  renderRoute(undefined, {
    history: '/session',
    routes: [
      {
        path: '/session',
        element: (
          <SessionDeleteButton
            workspace={DEFAULT_WORKSPACE}
            sessionId={SESSION}
            traceCount={2}
            spanCount={7}
          />
        ),
      },
      { path: getIntakeTracesRoute(DEFAULT_WORKSPACE), element: <div>Remaining trace runs</div> },
    ],
  });

const allow = () => server.use(http.get(PERMISSION, () => HttpResponse.json({ allowed: true })));

const confirm = async (retry = false) => {
  await userEvent.click(
    await screen.findByRole('button', {
      name: retry ? 'Retry session telemetry deletion' : 'Delete session telemetry',
    })
  );
  const dialog = await screen.findByRole('dialog');
  await userEvent.click(within(dialog).getByRole('button', { name: 'Delete Session Telemetry' }));
};

it.each([false, true])('gates deletion on server permission (allowed: %s)', async (allowed) => {
  const checked = vi.fn(() => HttpResponse.json({ allowed }));
  server.use(http.get(PERMISSION, checked));
  show();
  await waitFor(() => expect(checked).toHaveBeenCalledOnce());
  if (allowed)
    expect(await screen.findByRole('button', { name: 'Delete session telemetry' })).toBeVisible();
  else
    expect(
      screen.queryByRole('button', { name: 'Delete session telemetry' })
    ).not.toBeInTheDocument();
});

it('explains the whole-session scope and counts before confirmation, and allows cancellation', async () => {
  allow();
  const deleted = vi.fn(() => new HttpResponse(null, { status: 204 }));
  server.use(http.delete(DELETE, deleted));
  show();
  await userEvent.click(await screen.findByRole('button', { name: 'Delete session telemetry' }));
  const dialog = await screen.findByRole('dialog');
  expect(within(dialog).getByText(/ALL current traces and spans/)).toHaveTextContent(
    `Workspace: ${DEFAULT_WORKSPACE}. Session: ${SESSION}`
  );
  expect(within(dialog).getByText(/Traces: 2. Spans: 7/)).toHaveTextContent(
    'Annotations and evaluator results are retained'
  );
  await userEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));
  expect(deleted).not.toHaveBeenCalled();
});

it.each([204, 404])('refreshes and leaves the deleted detail on %s', async (status) => {
  allow();
  const targets: string[] = [];
  server.use(
    http.delete(DELETE, ({ request }) => {
      targets.push(new URL(request.url).pathname);
      return new HttpResponse(null, { status });
    })
  );
  show();
  await confirm();
  expect(await screen.findByText('Remaining trace runs')).toBeVisible();
  expect(targets).toEqual([`/apis/intake/v2/workspaces/${DEFAULT_WORKSPACE}/sessions/${SESSION}`]);
});

it.each([
  [403, 'Deletion requires intake.sessions.delete'],
  [409, 'Deletion is blocked by an evaluation association'],
  [503, 'Cleanup may be incomplete'],
])('keeps the target available and explains %s', async (status, message) => {
  allow();
  server.use(
    http.delete(DELETE, () =>
      HttpResponse.json({ detail: 'Backend explanation' }, { status: Number(status) })
    )
  );
  show();
  await confirm();
  expect(await screen.findByRole('alert')).toHaveTextContent(String(message));
  expect(screen.getByRole('alert')).toHaveTextContent('Backend explanation');
  expect(screen.queryByText('Remaining trace runs')).not.toBeInTheDocument();
  expect(
    await screen.findByRole('button', { name: 'Retry session telemetry deletion' })
  ).toBeVisible();
});

it('retries partial cleanup directly against the same target without fetching session detail', async () => {
  allow();
  let attempts = 0;
  const detail = vi.fn(() => new HttpResponse(null, { status: 404 }));
  server.use(
    http.get(DELETE, detail),
    http.delete(DELETE, () =>
      ++attempts === 1
        ? HttpResponse.json({ detail: 'storage unavailable' }, { status: 503 })
        : new HttpResponse(null, { status: 204 })
    )
  );
  show();
  await confirm();
  expect(await screen.findByRole('alert')).toHaveTextContent(
    'even if the session can no longer be loaded'
  );
  await confirm(true);
  expect(await screen.findByText('Remaining trace runs')).toBeVisible();
  expect(attempts).toBe(2);
  expect(detail).not.toHaveBeenCalled();
});
