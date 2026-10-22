// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_DEBOUNCE_MS } from '@nemo/common/src/constants';
import { useFilesListFilesets } from '@nemo/sdk/generated/platform/files';
import { FilesetNameField } from '@studio/components/filesets/FilesetNameField';
import type { NameCheckStatus } from '@studio/components/form/entityNameCheck';
import { act, render, screen } from '@testing-library/react';
import { type ReactElement, useEffect } from 'react';
import { useForm } from 'react-hook-form';

vi.mock('@nemo/sdk/generated/platform/files');

const mockListFilesets = vi.mocked(useFilesListFilesets);

/** The list check finds exactly the names in `taken`. */
const takenNames = (...taken: string[]) =>
  mockListFilesets.mockImplementation(
    (_workspace, params) =>
      ({
        data: { data: taken.some((name) => name === params?.filter?.name) ? [{}] : [] },
        isFetching: false,
        isError: false,
      }) as unknown as ReturnType<typeof useFilesListFilesets>
  );

interface HarnessProps {
  value?: string;
  schemaError?: string;
  ownName?: string;
  onStatusChange?: (status: NameCheckStatus) => void;
}

const Harness = ({
  value = '',
  schemaError,
  ownName,
  onStatusChange = vi.fn(),
}: HarnessProps): ReactElement => {
  const { control, setError } = useForm<{ name: string }>({ defaultValues: { name: value } });
  useEffect(() => {
    if (schemaError) setError('name', { message: schemaError });
  }, [schemaError, setError]);
  return (
    <FilesetNameField
      control={control}
      name="name"
      workspace="default"
      ownName={ownName}
      onStatusChange={onStatusChange}
    />
  );
};

/** Renders the field and lets the check's debounce settle. */
const renderSettled = (props: HarnessProps) => {
  render(<Harness {...props} />);
  act(() => {
    vi.advanceTimersByTime(DEFAULT_DEBOUNCE_MS);
  });
};

describe('FilesetNameField', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    mockListFilesets.mockReset();
    takenNames('taken');
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('says where the name must be unique before anything is typed', () => {
    render(<Harness />);

    expect(
      screen.getByText(
        (_, element) =>
          element?.getAttribute('data-testid') === 'nv-form-field-helper' &&
          element.textContent === 'Fileset names must be unique within the default workspace.'
      )
    ).toBeInTheDocument();
  });

  it('confirms a free name with its workspace', () => {
    renderSettled({ value: 'my-data' });

    expect(screen.getByText(/will be created as/)).toHaveTextContent(
      'Your fileset will be created as my-data in the default workspace'
    );
  });

  it('reports the check’s status to the form', () => {
    const onStatusChange = vi.fn();
    renderSettled({ value: 'taken', onStatusChange });

    expect(onStatusChange).toHaveBeenLastCalledWith('conflict');
  });

  it('does not count a fileset the form created itself as taken', () => {
    const onStatusChange = vi.fn();
    renderSettled({ value: 'taken', ownName: 'taken', onStatusChange });

    expect(onStatusChange).toHaveBeenLastCalledWith(undefined);
    expect(screen.queryByText(/already exists/)).not.toBeInTheDocument();
  });

  it('puts a conflict ahead of the field’s own format error', () => {
    renderSettled({ value: 'taken', schemaError: 'Name is too short.' });

    expect(
      screen.getByText('A fileset named taken already exists in the default workspace')
    ).toBeInTheDocument();
    expect(screen.queryByText('Name is too short.')).not.toBeInTheDocument();
  });

  it('shows the form’s own error when the name has not been checked', () => {
    render(<Harness schemaError="Name is required." />);

    expect(screen.getByText('Name is required.')).toBeInTheDocument();
  });
});
