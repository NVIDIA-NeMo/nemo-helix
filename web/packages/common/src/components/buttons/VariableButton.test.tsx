// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { VariableButton } from '@nemo/common/src/components/buttons/VariableButton';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const VARIABLES = [{ name: 'input', description: 'The dataset input.' }, { name: 'output' }];

describe('VariableButton', () => {
  it('renders the trigger with the Variable label', () => {
    render(<VariableButton variables={VARIABLES} onSelect={() => {}} />);
    expect(screen.getByRole('button', { name: /variable/i })).toBeInTheDocument();
  });

  it('opens a panel listing each variable on click', async () => {
    const user = userEvent.setup();
    render(<VariableButton variables={VARIABLES} onSelect={() => {}} />);
    await user.click(screen.getByRole('button', { name: /variable/i }));
    expect(await screen.findByRole('button', { name: /input/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /output/ })).toBeInTheDocument();
  });

  it('calls onSelect with the chosen variable', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(<VariableButton variables={VARIABLES} onSelect={onSelect} />);
    await user.click(screen.getByRole('button', { name: /variable/i }));
    await user.click(await screen.findByRole('button', { name: /input/ }));
    expect(onSelect).toHaveBeenCalledWith({ name: 'input', description: 'The dataset input.' });
  });

  it('disables the trigger when no variables are available', () => {
    render(<VariableButton variables={[]} onSelect={() => {}} />);
    expect(screen.getByRole('button', { name: /variable/i })).toBeDisabled();
  });

  it('respects the disabled prop', () => {
    render(<VariableButton variables={VARIABLES} onSelect={() => {}} disabled />);
    expect(screen.getByRole('button', { name: /variable/i })).toBeDisabled();
  });
});
