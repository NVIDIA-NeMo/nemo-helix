// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ScoreModal } from '@studio/components/evaluation/Jobs/form/ScoreModal';
import { renderRoute, screen } from '@studio/tests/util/render';
import { waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const renderScoreModal = (onSave = vi.fn()) => {
  const onClose = vi.fn();

  renderRoute(<ScoreModal open onClose={onClose} onSave={onSave} />);

  return { onClose, onSave };
};

describe('ScoreModal', () => {
  it('preserves score type fields when switching between rubric and range', async () => {
    const user = userEvent.setup();
    const { onSave } = renderScoreModal();

    await user.type(screen.getByPlaceholderText('e.g., quality'), 'quality');
    await user.type(
      screen.getByRole('textbox', { name: 'Scoring Criteria' }),
      'How good the answer is'
    );

    const rubricLabelInputs = screen.getAllByPlaceholderText('Label');
    const rubricDescriptionInputs = screen.getAllByPlaceholderText('Description');
    const rubricValueInputs = screen.getAllByRole('spinbutton');

    await user.type(rubricLabelInputs[0], 'fails');
    await user.type(rubricDescriptionInputs[0], 'Does not answer');
    await user.type(rubricValueInputs[0], '0');
    await user.type(rubricLabelInputs[1], 'excellent');
    await user.type(rubricDescriptionInputs[1], 'Fully answers');
    await user.type(rubricValueInputs[1], '4');

    await user.click(screen.getByRole('radio', { name: 'Range' }));

    const rangeInputs = screen.getAllByRole('spinbutton');
    await user.clear(rangeInputs[0]);
    await user.type(rangeInputs[0], '0');
    await user.clear(rangeInputs[1]);
    await user.type(rangeInputs[1], '10');

    await user.click(screen.getByRole('radio', { name: 'Rubric' }));

    expect(screen.getAllByPlaceholderText('Label')[0]).toHaveValue('fails');
    expect(screen.getAllByPlaceholderText('Description')[0]).toHaveValue('Does not answer');
    expect(screen.getAllByRole('spinbutton')[0]).toHaveValue(0);
    expect(screen.getAllByPlaceholderText('Label')[1]).toHaveValue('excellent');
    expect(screen.getAllByPlaceholderText('Description')[1]).toHaveValue('Fully answers');
    expect(screen.getAllByRole('spinbutton')[1]).toHaveValue(4);

    await user.click(screen.getByRole('radio', { name: 'Range' }));

    expect(screen.getAllByRole('spinbutton')[0]).toHaveValue(0);
    expect(screen.getAllByRole('spinbutton')[1]).toHaveValue(10);

    await user.click(screen.getByRole('button', { name: 'Add Score' }));

    await waitFor(() => {
      expect(onSave).toHaveBeenCalledWith(
        expect.objectContaining({
          scoreType: 'range',
          name: 'quality',
          minimum: 0,
          maximum: 10,
        })
      );
    });
  });
});
