// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledDatasetFileSelect } from '@nemo/common/src/components/DatasetFileSelect/ControlledDatasetFileSelect';
import type { FileListItem } from '@nemo/common/src/components/FileList';
import { render } from '@testing-library/react';
import { act, type FC } from 'react';
import { useForm } from 'react-hook-form';

// Capture props passed to DatasetFileSelect so tests can call onChange and inspect value
let capturedOnChange: ((files: FileListItem[]) => void) | null = null;
let capturedValue: FileListItem | FileListItem[] | null | undefined = undefined;

vi.mock('@nemo/common/src/components/DatasetFileSelect/DatasetFileSelect', () => ({
  DatasetFileSelect: vi.fn((props) => {
    capturedOnChange = props.onChange;
    capturedValue = props.value;
    return <div data-testid="dataset-file-select" />;
  }),
}));

vi.mock('@nvidia/foundations-react-core', () => ({
  FormField: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

const Wrapper: FC<{ defaultValue?: string | null }> = ({ defaultValue = null }) => {
  const { control } = useForm<{ dataset: string | null }>({
    defaultValues: { dataset: defaultValue },
  });
  return (
    <ControlledDatasetFileSelect
      useControllerProps={{ name: 'dataset', control }}
      workspace="test-ws"
    />
  );
};

describe('ControlledDatasetFileSelect', () => {
  describe('handleChange — converts selection to workspace/name#path', () => {
    it('outputs workspace/name#path when a file with a fileset:// url is selected', () => {
      render(<Wrapper />);

      act(() => {
        capturedOnChange!([{ path: 'train.jsonl', url: 'fileset://my-ws/my-ds/train.jsonl' }]);
      });

      // After onChange, RHF re-renders and passes the stored workspace/name#path back as url.
      expect((capturedValue as FileListItem)?.path).toBe('train.jsonl');
      expect((capturedValue as FileListItem)?.url).toBe('my-ws/my-ds#train.jsonl');
    });

    it('outputs workspace/name#nested/path for deep paths', () => {
      render(<Wrapper />);

      act(() => {
        capturedOnChange!([
          { path: 'data/sub/file.csv', url: 'fileset://ws/ds/data/sub/file.csv' },
        ]);
      });

      expect((capturedValue as FileListItem)?.path).toBe('data/sub/file.csv');
      expect((capturedValue as FileListItem)?.url).toBe('ws/ds#data/sub/file.csv');
    });

    it('sets value to null when files are cleared', () => {
      render(<Wrapper defaultValue="ws/ds#file.jsonl" />);

      act(() => {
        capturedOnChange!([]);
      });

      expect(capturedValue).toBeNull();
    });

    it('sets value to null when the selected file has no parseable fileset:// url', () => {
      render(<Wrapper />);

      act(() => {
        capturedOnChange!([{ path: 'file.jsonl', url: 'blob:http://localhost/some-blob' }]);
      });

      expect(capturedValue).toBeNull();
    });
  });

  describe('selectedFile — converts workspace/name#path back to FileListItem', () => {
    it('converts workspace/name#path to FileListItem with url set to the form value', () => {
      render(<Wrapper defaultValue="my-ws/my-ds#train.jsonl" />);
      expect((capturedValue as FileListItem)?.path).toBe('train.jsonl');
      expect((capturedValue as FileListItem)?.url).toBe('my-ws/my-ds#train.jsonl');
    });

    it('handles nested paths', () => {
      render(<Wrapper defaultValue="ws/ds#folder/sub/file.csv" />);
      expect((capturedValue as FileListItem)?.path).toBe('folder/sub/file.csv');
      expect((capturedValue as FileListItem)?.url).toBe('ws/ds#folder/sub/file.csv');
    });

    it('returns null for empty value', () => {
      render(<Wrapper defaultValue={null} />);
      expect(capturedValue).toBeNull();
    });

    it('returns null for a root-only ref with no file path', () => {
      render(<Wrapper defaultValue="ws/ds" />);
      expect(capturedValue).toBeNull();
    });
  });
});
