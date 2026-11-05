// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { InferenceParameters } from '@nemo/common/src/components/ModelSelectV2/InferenceParameters';
import type { InferenceParams } from '@nemo/sdk/generated/platform/schema';
import { fireEvent, render, screen } from '@testing-library/react';
import { useState } from 'react';

const renderComponent = (props: Partial<React.ComponentProps<typeof InferenceParameters>> = {}) => {
  const onChange = vi.fn();
  render(<InferenceParameters value={{}} onChange={onChange} {...props} />);
  return { onChange };
};

describe('InferenceParameters', () => {
  it('renders temperature, max tokens, and top p fields', () => {
    renderComponent();
    expect(screen.getByText('Temperature')).toBeInTheDocument();
    expect(screen.getByText('Max Tokens')).toBeInTheDocument();
    expect(screen.getByText('Top P')).toBeInTheDocument();
  });

  it('uses 1024 as the default for max tokens', () => {
    renderComponent();
    const inputs = screen.getAllByRole('spinbutton');
    // inputs: [temperature, max_tokens, top_p]
    expect(inputs[1]).toHaveValue(1024);
  });

  it('displays provided initial values', () => {
    renderComponent({ value: { temperature: 0.5, max_tokens: 512, top_p: 0.8 } });
    const inputs = screen.getAllByRole('spinbutton');
    expect(inputs[0]).toHaveValue(0.5);
    expect(inputs[1]).toHaveValue(512);
    expect(inputs[2]).toHaveValue(0.8);
  });

  it('calls onChange with updated temperature and preserves other fields', () => {
    const { onChange } = renderComponent({ value: { max_tokens: 512, top_p: 0.9 } });
    const inputs = screen.getAllByRole('spinbutton');
    fireEvent.change(inputs[0], { target: { value: '0.7' } });
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ temperature: 0.7, max_tokens: 512, top_p: 0.9 })
    );
  });

  it('calls onChange with updated max_tokens only (not max_completion_tokens)', () => {
    const { onChange } = renderComponent({ value: { temperature: 0.5 } });
    const inputs = screen.getAllByRole('spinbutton');
    fireEvent.change(inputs[1], { target: { value: '256' } });
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ max_tokens: 256, temperature: 0.5 })
    );
    expect(onChange).not.toHaveBeenCalledWith(
      expect.objectContaining({ max_completion_tokens: expect.anything() })
    );
  });

  it('calls onChange with updated top_p and preserves other fields', () => {
    const { onChange } = renderComponent({ value: { temperature: 0.5, max_tokens: 256 } });
    const inputs = screen.getAllByRole('spinbutton');
    fireEvent.change(inputs[2], { target: { value: '0.9' } });
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ top_p: 0.9, temperature: 0.5, max_tokens: 256 })
    );
  });

  it('lets the user clear the temperature field and type a new value', () => {
    const StatefulHarness = () => {
      const [value, setValue] = useState<Partial<InferenceParams>>({ temperature: 1 });
      return <InferenceParameters value={value} onChange={setValue} />;
    };
    render(<StatefulHarness />);
    const temperature = screen.getAllByRole('spinbutton')[0];

    fireEvent.change(temperature, { target: { value: '' } });
    expect(temperature).toHaveValue(null);

    fireEvent.change(temperature, { target: { value: '0.7' } });
    expect(temperature).toHaveValue(0.7);
  });

  it('renders only the requested fields, in the given order', () => {
    renderComponent({ fields: ['temperature', 'max_tokens'] });
    expect(screen.getByText('Temperature')).toBeInTheDocument();
    expect(screen.getByText('Max Tokens')).toBeInTheDocument();
    expect(screen.queryByText('Top P')).not.toBeInTheDocument();
    expect(screen.getAllByRole('spinbutton')).toHaveLength(2);
  });

  it('applies per-field metadata overrides to defaults and bounds', () => {
    const { onChange } = renderComponent({
      fieldMetadata: { max_tokens: { default: 512, min: 32, max: 4096 } },
    });
    const maxTokens = screen.getAllByRole('spinbutton')[1];
    expect(maxTokens).toHaveValue(512);
    fireEvent.change(maxTokens, { target: { value: '8192' } });
    fireEvent.blur(maxTokens);
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ max_tokens: 4096 }));
  });

  it('names each slider, number input, and reset button after its field', () => {
    renderComponent();
    expect(screen.getByRole('slider', { name: 'Temperature' })).toBeInTheDocument();
    expect(screen.getByRole('spinbutton', { name: 'Max Tokens value' })).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: 'Reset Top P to default value' })
    ).toBeInTheDocument();
  });

  it('disables all inputs when disabled', () => {
    renderComponent({ disabled: true });
    const inputs = screen.getAllByRole('spinbutton');
    inputs.forEach((input) => expect(input).toBeDisabled());
  });
});
