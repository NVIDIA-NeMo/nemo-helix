// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FinetuningTypeFilterField } from '@studio/components/dataViews/CustomModelsDataView/FinetuningTypeFilterField';
import { act, render } from '@testing-library/react';

interface ComboboxStubProps {
  value: string[];
  onOpenChange: (open: boolean) => void;
  onValueChange: (value: string[]) => void;
}

const combobox = vi.hoisted(() => ({ props: undefined as ComboboxStubProps | undefined }));

vi.mock('@nvidia/foundations-react-core', () => ({
  Combobox: (props: ComboboxStubProps) => {
    combobox.props = props;
    return null;
  },
}));

const props = () => combobox.props as ComboboxStubProps;

describe('FinetuningTypeFilterField', () => {
  it('commits picks once when the listbox closes', () => {
    const onValueChange = vi.fn();
    render(<FinetuningTypeFilterField value={[]} onValueChange={onValueChange} />);

    act(() => props().onOpenChange(true));
    act(() => props().onValueChange(['lora']));
    act(() => props().onValueChange(['lora', 'dpo']));
    expect(onValueChange).not.toHaveBeenCalled();
    expect(props().value).toEqual(['lora', 'dpo']);

    act(() => props().onOpenChange(false));
    expect(onValueChange).toHaveBeenCalledTimes(1);
    expect(onValueChange).toHaveBeenCalledWith(['lora', 'dpo']);
  });

  it('skips the commit when nothing changed', () => {
    const onValueChange = vi.fn();
    render(<FinetuningTypeFilterField value={['dpo']} onValueChange={onValueChange} />);

    act(() => props().onOpenChange(true));
    act(() => props().onOpenChange(false));
    expect(onValueChange).not.toHaveBeenCalled();
  });

  it('commits immediately when cleared while closed', () => {
    const onValueChange = vi.fn();
    render(<FinetuningTypeFilterField value={['dpo']} onValueChange={onValueChange} />);

    act(() => props().onValueChange([]));
    expect(onValueChange).toHaveBeenCalledWith([]);
  });

  it('commits pending picks on unmount', () => {
    const onValueChange = vi.fn();
    const { unmount } = render(
      <FinetuningTypeFilterField value={[]} onValueChange={onValueChange} />
    );

    act(() => props().onOpenChange(true));
    act(() => props().onValueChange(['grpo']));
    unmount();
    expect(onValueChange).toHaveBeenCalledWith(['grpo']);
  });

  it('syncs to external value changes', () => {
    const { rerender } = render(
      <FinetuningTypeFilterField value={['dpo']} onValueChange={vi.fn()} />
    );
    rerender(<FinetuningTypeFilterField value={[]} onValueChange={vi.fn()} />);
    expect(props().value).toEqual([]);
  });
});
