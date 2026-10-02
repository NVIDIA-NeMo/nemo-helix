// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTES } from '@studio/constants/routes';
import { matchRoutes } from 'react-router';

/**
 * `/fine-tune/new` and `/fine-tune/:customizationJobName` are siblings, so the create page
 * is only reachable while the static segment out-ranks the parameter. React Router does
 * rank it higher, but nothing in the route table says so — this is what would catch a
 * reordering, or a rename that turned `new` into a job called "new".
 */
describe('fine-tuning routes', () => {
  const table = [
    { path: ROUTES.workspace.customizationJobList, id: 'list' },
    { path: ROUTES.workspace.newCustomizationJob, id: 'new' },
    { path: ROUTES.workspace.customizationJobDetails, id: 'details' },
  ];

  const matchedId = (pathname: string) => matchRoutes(table, pathname)?.at(-1)?.route.id;

  it('sends /fine-tune/new to the create page, not to a job named "new"', () => {
    expect(matchedId('/workspaces/default/fine-tune/new')).toBe('new');
  });

  it('sends a job name to the details page', () => {
    expect(matchedId('/workspaces/default/fine-tune/my-job')).toBe('details');
  });

  it('sends the bare path to the list', () => {
    expect(matchedId('/workspaces/default/fine-tune')).toBe('list');
  });
});
