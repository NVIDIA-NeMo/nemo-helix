// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ParamsDropdown } from '@nemo/common/src/components/ModelSelectV2/ParamsDropdown';
import { fireEvent, render, screen } from '@testing-library/react';

const renderOpen = (props: Partial<React.ComponentProps<typeof ParamsDropdown>> = {}) => {
  const onOpenChange = vi.fn();
  const onInferenceParamsChange = vi.fn();
  render(
    <ParamsDropdown
      open
      onOpenChange={onOpenChange}
      onInferenceParamsChange={onInferenceParamsChange}
      {...props}
    />
  );
  return { onOpenChange, onInferenceParamsChange };
};

describe('ParamsDropdown', () => {
  it('renders the trigger button', () => {
    render(<ParamsDropdown open={false} onOpenChange={vi.fn()} />);
    expect(screen.getByRole('button', { name: 'Model parameters' })).toBeInTheDocument();
  });

  it('opens on its own when the caller does not control it', async () => {
    render(<ParamsDropdown />);
    const trigger = screen.getByTestId('params-dropdown-trigger');
    fireEvent.pointerDown(trigger, { button: 0 });
    fireEvent.click(trigger);
    expect(await screen.findByTestId('params-dropdown-content')).toBeInTheDocument();
  });

  it('shows only the requested fields', async () => {
    render(<ParamsDropdown open fields={['temperature', 'max_tokens']} />);
    await screen.findByText('Temperature');
    expect(screen.queryByText('Top P')).not.toBeInTheDocument();
  });

  describe('when open', () => {
    it('renders temperature, max tokens, and top p sliders', async () => {
      renderOpen();
      expect(await screen.findByText('Temperature')).toBeInTheDocument();
      expect(await screen.findByText('Max Tokens')).toBeInTheDocument();
      expect(await screen.findByText('Top P')).toBeInTheDocument();
    });

    it('uses 1024 as the default for max tokens', async () => {
      renderOpen();
      await screen.findByText('Max Tokens');
      const inputs = screen.getAllByRole('spinbutton');
      expect(inputs[1]).toHaveValue(1024);
    });

    it('calls onInferenceParamsChange with updated temperature', async () => {
      const { onInferenceParamsChange } = renderOpen();
      await screen.findByText('Temperature');
      const [temperatureInput] = screen.getAllByRole('spinbutton');
      fireEvent.change(temperatureInput, { target: { value: '0.7' } });
      expect(onInferenceParamsChange).toHaveBeenCalledWith(
        expect.objectContaining({ temperature: 0.7 })
      );
    });

    it('calls onInferenceParamsChange with only max_tokens when max tokens input changes', async () => {
      const { onInferenceParamsChange } = renderOpen();
      await screen.findByText('Max Tokens');
      const inputs = screen.getAllByRole('spinbutton');
      fireEvent.change(inputs[1], { target: { value: '512' } });
      expect(onInferenceParamsChange).toHaveBeenCalledWith(
        expect.objectContaining({ max_tokens: 512 })
      );
      expect(onInferenceParamsChange).not.toHaveBeenCalledWith(
        expect.objectContaining({ max_completion_tokens: expect.anything() })
      );
    });

    it('calls onInferenceParamsChange with updated top_p', async () => {
      const { onInferenceParamsChange } = renderOpen();
      await screen.findByText('Top P');
      const inputs = screen.getAllByRole('spinbutton');
      fireEvent.change(inputs[2], { target: { value: '0.9' } });
      expect(onInferenceParamsChange).toHaveBeenCalledWith(expect.objectContaining({ top_p: 0.9 }));
    });

    it('preserves existing params when a single field changes', async () => {
      const { onInferenceParamsChange } = renderOpen({
        inferenceParams: { temperature: 0.5, top_p: 0.8 },
      });
      await screen.findByText('Temperature');
      const inputs = screen.getAllByRole('spinbutton');
      fireEvent.change(inputs[1], { target: { value: '256' } });
      expect(onInferenceParamsChange).toHaveBeenCalledWith(
        expect.objectContaining({ temperature: 0.5, top_p: 0.8, max_tokens: 256 })
      );
    });

    it('displays inferenceParams initial values', async () => {
      renderOpen({ inferenceParams: { temperature: 0.5, max_tokens: 256, top_p: 0.8 } });
      await screen.findByText('Temperature');
      const inputs = screen.getAllByRole('spinbutton');
      expect(inputs[0]).toHaveValue(0.5);
      expect(inputs[1]).toHaveValue(256);
      expect(inputs[2]).toHaveValue(0.8);
    });

    it('disables all slider inputs when disabled', async () => {
      renderOpen({ disabled: true });
      await screen.findByText('Temperature');
      const inputs = screen.getAllByRole('spinbutton');
      inputs.forEach((input) => expect(input).toBeDisabled());
    });
  });
});
