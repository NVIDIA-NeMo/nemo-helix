// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';

import { MarkdownContent } from '.';

describe('MarkdownContent', () => {
  describe('basic rendering', () => {
    it('renders plain prose text', () => {
      render(<MarkdownContent content="Hello world" />);
      expect(screen.getByText('Hello world')).toBeInTheDocument();
    });

    it('renders headings as semantic h1/h2 elements', () => {
      render(<MarkdownContent content={'# Big Title\n\n## Subtitle'} />);
      expect(screen.getByRole('heading', { level: 1, name: 'Big Title' })).toBeInTheDocument();
      expect(screen.getByRole('heading', { level: 2, name: 'Subtitle' })).toBeInTheDocument();
    });

    it('supports GFM tables (proves remark-gfm is wired)', () => {
      const content = '| Col A | Col B |\n| --- | --- |\n| a1 | b1 |';
      render(<MarkdownContent content={content} />);
      expect(screen.getByRole('columnheader', { name: 'Col A' })).toBeInTheDocument();
      expect(screen.getByRole('cell', { name: 'a1' })).toBeInTheDocument();
    });
  });

  describe('links', () => {
    it('opens links in a new tab with rel="noopener noreferrer"', () => {
      render(<MarkdownContent content="See [the docs](https://example.com)." />);
      const link = screen.getByRole('link', { name: 'the docs' });
      expect(link).toHaveAttribute('href', 'https://example.com');
      expect(link).toHaveAttribute('target', '_blank');
      expect(link.getAttribute('rel')).toMatch(/noopener/);
      expect(link.getAttribute('rel')).toMatch(/noreferrer/);
    });
  });

  describe('code blocks', () => {
    it('uses light and dark grey backgrounds for fenced code', () => {
      render(<MarkdownContent content={'```typescript\nconst value = 1;\n```'} />);

      expect(screen.getByTestId('nv-code-snippet-code')).toHaveClass(
        '[&&]:bg-gray-050',
        'dark:[&&]:bg-gray-900'
      );
    });

    it('uses light and dark grey backgrounds with prose font for inline code', () => {
      render(<MarkdownContent content="Use `const value = 1` inline." />);

      expect(screen.getByTestId('nv-code-snippet-code')).toHaveClass(
        '[&&]:bg-gray-050',
        'dark:[&&]:bg-gray-900',
        '[&&]:rounded',
        '[&&]:font-sans'
      );
    });
  });

  describe('blockquotes', () => {
    it('renders a plain blockquote when no callout marker is present', () => {
      render(<MarkdownContent content="> A regular quote." />);
      expect(screen.getByRole('blockquote')).toHaveTextContent('A regular quote.');
      // No callout label rendered.
      expect(screen.queryByText(/^(Note|Tip|Important|Warning|Caution)$/)).not.toBeInTheDocument();
    });

    it('does not treat a marker that appears later in the body as a callout', () => {
      render(<MarkdownContent content="> Some text and then [!NOTE] in the middle." />);
      // No callout label — marker is not at the start of the first paragraph.
      expect(screen.queryByText('Note')).not.toBeInTheDocument();
      // The literal marker text is preserved in the rendered prose.
      expect(screen.getByText(/Some text and then \[!NOTE\] in the middle/)).toBeInTheDocument();
    });
  });

  describe('callouts', () => {
    const cases = [
      { kind: 'NOTE', label: 'Note' },
      { kind: 'TIP', label: 'Tip' },
      { kind: 'IMPORTANT', label: 'Important' },
      { kind: 'WARNING', label: 'Warning' },
      { kind: 'CAUTION', label: 'Caution' },
    ];

    it.each(cases)(
      'renders [!$kind] as a $label callout with the marker stripped from the body',
      ({ kind, label }) => {
        const body = `${label} body text.`;
        render(<MarkdownContent content={`> [!${kind}]\n> ${body}`} />);

        // Foundations-styled label appears.
        expect(screen.getByText(label)).toBeInTheDocument();
        // Body content is rendered after marker stripping.
        expect(screen.getByText(body)).toBeInTheDocument();
        // The `[!KIND]` marker itself is not part of the rendered text.
        expect(screen.queryByText(new RegExp(`\\[!${kind}\\]`))).not.toBeInTheDocument();
      }
    );
  });

  describe('images', () => {
    it('renders a markdown image by default', () => {
      render(<MarkdownContent content="![alt text](https://example.com/pic.png)" />);
      const img = screen.getByRole('img', { name: 'alt text' });
      expect(img).toHaveAttribute('src', 'https://example.com/pic.png');
    });

    it('drops markdown images when disableImages is set (untrusted content, CWE-200)', () => {
      render(
        <MarkdownContent
          content="before ![alt text](https://attacker.example/track.png) after"
          disableImages
        />
      );
      // No <img> is emitted, so the viewer's browser never fetches the remote src.
      expect(screen.queryByRole('img')).not.toBeInTheDocument();
      // Surrounding markdown still renders.
      expect(screen.getByText(/before/)).toBeInTheDocument();
      expect(screen.getByText(/after/)).toBeInTheDocument();
    });
  });
});
