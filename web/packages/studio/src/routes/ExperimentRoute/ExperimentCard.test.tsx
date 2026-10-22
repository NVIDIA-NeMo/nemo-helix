// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import type { ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { ExperimentCard } from '@studio/routes/ExperimentRoute/ExperimentCard';
import { render, screen } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

it.each([false, true])('offers deletion on experiment cards (favorite: %s)', async (isFavorite) => {
  const group: ExperimentResponse = {
    id: 'experiment',
    name: 'sample',
    workspace: DEFAULT_WORKSPACE,
    is_favorite: isFavorite,
    default_sort: '-created_at',
  };
  render(<ExperimentCard group={group} workspace={DEFAULT_WORKSPACE} />);
  expect(screen.getByRole('link', { name: 'sample' })).toBeVisible();
  await userEvent.click(screen.getByRole('button', { name: 'Delete experiment sample' }));
  expect(await screen.findByRole('dialog')).toBeVisible();
  expect(
    screen.getByText(/Evaluations belonging only to the selected experiments will also be deleted/)
  ).toBeVisible();
});
