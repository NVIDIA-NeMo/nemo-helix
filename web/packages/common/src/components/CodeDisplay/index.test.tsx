// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CodeDisplay } from '@nemo/common/src/components/CodeDisplay';
import { render, screen } from '@testing-library/react';

const multiLineString = `
  typescript
  console.log('Testing');
  console.log('Testing Next Line');
`;

describe('CodeDisplay', () => {
  it('should extract language identifier from code content and display it in the slotActions', () => {
    render(<CodeDisplay>{multiLineString}</CodeDisplay>);
    const codeActions = screen.getByTestId('nv-code-snippet-actions');

    expect(codeActions.textContent).toContain('typescript');
  });

  it('should remove language identifier from displayed code', () => {
    render(<CodeDisplay>{multiLineString}</CodeDisplay>);

    const codeContent = screen.getByTestId('nv-code-snippet-code');
    expect(codeContent.textContent).not.toContain('typescript');
  });

  it('should pass extracted language to CodeSnippet', () => {
    render(<CodeDisplay>{multiLineString}</CodeDisplay>);

    const codeContent = screen.getByTestId('nv-code-snippet-code');
    expect(codeContent.textContent).toContain("console.log('Testing')");
    expect(codeContent.textContent).toContain("console.log('Testing Next Line')");
  });

  it('uses the chat code block light and dark grey backgrounds', () => {
    render(<CodeDisplay>{multiLineString}</CodeDisplay>);

    expect(screen.getByTestId('code-display')).toHaveClass('my-density-md');
    expect(screen.getByTestId('nv-code-snippet-code')).toHaveClass(
      '!bg-gray-050',
      '!py-density-xs',
      'dark:!bg-gray-900'
    );
  });
});
