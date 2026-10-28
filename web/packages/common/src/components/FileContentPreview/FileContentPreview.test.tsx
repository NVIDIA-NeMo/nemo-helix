// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FileContentPreview } from '@nemo/common/src/components/FileContentPreview/index';
import { render, screen } from '@testing-library/react';

vi.mock('@nemo/common/src/components/CodeEditor', () => ({
  CodeEditor: ({ content, contentType }: { content: string; contentType: string }) => (
    <div data-testid="code-editor" data-content-type={contentType}>
      {content}
    </div>
  ),
}));

vi.mock('@nemo/common/src/components/MarkdownContent', () => ({
  MarkdownContent: ({ content }: { content: string }) => (
    <div data-testid="markdown-content">{content}</div>
  ),
}));

// Mock papaparse
vi.mock('papaparse', () => ({
  default: {
    parse: vi.fn((content: string) => {
      const lines = content.trim().split('\n');
      if (lines.length === 0) {
        return { data: [], meta: { fields: [] }, errors: [] };
      }
      const headers = lines[0].split(',');
      const data = lines.slice(1).map((line) => {
        const values = line.split(',');
        const row: Record<string, string> = {};
        headers.forEach((header, index) => {
          row[header] = values[index] || '';
        });
        return row;
      });
      return { data, meta: { fields: headers }, errors: [] };
    }),
  },
}));

describe('FileContentPreview', () => {
  const defaultFile = { path: 'test.txt', url: '' };

  describe('loading / error / empty states', () => {
    it('renders spinner when loading', () => {
      render(<FileContentPreview file={defaultFile} isLoading error={null} content={undefined} />);
      expect(screen.getByLabelText('Loading...')).toBeInTheDocument();
    });

    it('renders error message when error is provided', () => {
      render(
        <FileContentPreview
          file={defaultFile}
          isLoading={false}
          error={new Error('Failed to fetch file')}
        />
      );
      expect(screen.getByText('Error: Failed to fetch file')).toBeInTheDocument();
    });

    it('renders fallback message when error has no message', () => {
      const error = new Error();
      error.message = '';
      render(<FileContentPreview file={defaultFile} isLoading={false} error={error} />);
      expect(screen.getByText('Error: Failed to load file')).toBeInTheDocument();
    });

    it('renders "No content available" when content is missing or empty', () => {
      const { rerender } = render(
        <FileContentPreview file={defaultFile} isLoading={false} error={null} content={undefined} />
      );
      expect(screen.getByText('No content available')).toBeInTheDocument();

      rerender(<FileContentPreview file={defaultFile} isLoading={false} error={null} content="" />);
      expect(screen.getByText('No content available')).toBeInTheDocument();
    });
  });

  describe('JSON / JSONL dispatch', () => {
    it('routes .json through CodeEditor with contentType=json', async () => {
      render(
        <FileContentPreview
          file={{ path: 'data.json' }}
          isLoading={false}
          error={null}
          content='{"key": "value"}'
        />
      );
      const editor = await screen.findByTestId('code-editor');
      expect(editor).toHaveAttribute('data-content-type', 'json');
      expect(editor).toHaveTextContent('{"key": "value"}');
    });

    it('routes .jsonl through CodeEditor with contentType=jsonl', async () => {
      render(
        <FileContentPreview
          file={{ path: 'data.jsonl' }}
          isLoading={false}
          error={null}
          content={'{"line": 1}\n{"line": 2}'}
        />
      );
      const editor = await screen.findByTestId('code-editor');
      expect(editor).toHaveAttribute('data-content-type', 'jsonl');
      expect(editor).toHaveTextContent('{"line": 1}');
    });

    it('handles nested file paths', async () => {
      render(
        <FileContentPreview
          file={{ path: 'folder/subfolder/data.json' }}
          isLoading={false}
          error={null}
          content='{"nested": true}'
        />
      );
      expect(await screen.findByTestId('code-editor')).toHaveAttribute('data-content-type', 'json');
    });
  });

  describe('Markdown dispatch', () => {
    it('routes .md through MarkdownContent', () => {
      render(
        <FileContentPreview
          file={{ path: 'README.md' }}
          isLoading={false}
          error={null}
          content="# Heading"
        />
      );
      const md = screen.getByTestId('markdown-content');
      expect(md).toHaveTextContent('# Heading');
      expect(screen.queryByTestId('code-editor')).toBeNull();
    });

    it('routes .markdown through MarkdownContent', () => {
      render(
        <FileContentPreview
          file={{ path: 'notes.markdown' }}
          isLoading={false}
          error={null}
          content="text"
        />
      );
      expect(screen.getByTestId('markdown-content')).toHaveTextContent('text');
    });

    it('treats an uppercase .MD extension as markdown', () => {
      render(
        <FileContentPreview
          file={{ path: 'README.MD' }}
          isLoading={false}
          error={null}
          content="# Heading"
        />
      );
      expect(screen.getByTestId('markdown-content')).toHaveTextContent('# Heading');
    });
  });

  describe('CSV files', () => {
    it('renders CSV content in a table', () => {
      render(
        <FileContentPreview
          file={{ path: 'data.csv' }}
          isLoading={false}
          error={null}
          content={'name,age\nAlice,30\nBob,25'}
        />
      );
      expect(screen.getByText('name')).toBeInTheDocument();
      expect(screen.getByText('age')).toBeInTheDocument();
      expect(screen.getByText('Alice')).toBeInTheDocument();
      expect(screen.getByText('30')).toBeInTheDocument();
    });

    it('treats an uppercase .CSV extension as CSV', () => {
      render(
        <FileContentPreview
          file={{ path: 'DATA.CSV' }}
          isLoading={false}
          error={null}
          content={'name,age\nAlice,30'}
        />
      );
      expect(screen.getByText('Alice')).toBeInTheDocument();
      expect(screen.queryByTestId('code-editor')).toBeNull();
    });
  });

  describe('Plain text fallback', () => {
    it('routes unknown extensions through CodeEditor with contentType=text', async () => {
      render(
        <FileContentPreview
          file={{ path: 'readme.txt' }}
          isLoading={false}
          error={null}
          content="This is plain text content"
        />
      );
      const editor = await screen.findByTestId('code-editor');
      expect(editor).toHaveAttribute('data-content-type', 'text');
      expect(editor).toHaveTextContent('This is plain text content');
    });
  });
});
