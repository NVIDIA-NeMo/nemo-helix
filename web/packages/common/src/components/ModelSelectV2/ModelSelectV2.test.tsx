// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ModelSelectV2 } from '@nemo/common/src/components/ModelSelectV2/ModelSelectV2';
import { fireEvent, render, screen } from '@testing-library/react';

const renderSelect = (props: Partial<React.ComponentProps<typeof ModelSelectV2>> = {}) => {
  const onInferenceParamsChange = vi.fn();
  render(
    <ModelSelectV2
      value={null}
      onValueChange={vi.fn()}
      groups={[]}
      showParams
      onInferenceParamsChange={onInferenceParamsChange}
      {...props}
    />
  );
  return { onInferenceParamsChange };
};

const openParams = async () => {
  fireEvent.pointerDown(screen.getByTestId('params-dropdown-trigger'), { button: 0 });
  fireEvent.click(screen.getByTestId('params-dropdown-trigger'));
  await screen.findByTestId('params-dropdown-content');
};

describe('ModelSelectV2', () => {
  it('renders no params control without showParams', () => {
    renderSelect({ showParams: false });
    expect(screen.queryByTestId('params-dropdown-trigger')).not.toBeInTheDocument();
  });

  it('disables the params control along with the model by default', () => {
    renderSelect({ disabled: true });
    expect(screen.getByTestId('model-select-v2-trigger')).toBeDisabled();
    expect(screen.getByTestId('params-dropdown-trigger')).toBeDisabled();
  });

  it('keeps params editable while the model is disabled when paramsDisabled is false', () => {
    renderSelect({ disabled: true, paramsDisabled: false });
    expect(screen.getByTestId('model-select-v2-trigger')).toBeDisabled();
    expect(screen.getByTestId('params-dropdown-trigger')).toBeEnabled();
  });

  it('passes the field subset and metadata through to the params editor', async () => {
    const { onInferenceParamsChange } = renderSelect({
      paramsFields: ['temperature', 'max_tokens'],
      paramsFieldMetadata: { temperature: { default: 0.7 } },
    });
    await openParams();

    expect(screen.queryByText('Top P')).not.toBeInTheDocument();
    const [temperature] = screen.getAllByRole('spinbutton');
    expect(temperature).toHaveValue(0.7);

    fireEvent.change(temperature, { target: { value: '0.4' } });
    expect(onInferenceParamsChange).toHaveBeenCalledWith({ temperature: 0.4 });
  });
});
