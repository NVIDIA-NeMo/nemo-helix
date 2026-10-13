// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MockToastProvider } from '@nemo/common/src/tests/MockToastProvider';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { type GenerateJob } from '@nemo/sdk/generated/safe-synthesizer/schema';
import { ThemeProvider } from '@nvidia/foundations-react-core';
import { JobConfigDrawer } from '@studio/routes/SafeSynthesizerJobDetailsRoute/components/JobConfigDrawer';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock the CodeEditor component to simplify testing
vi.mock('@nemo/common/src/components/CodeEditor', () => ({
  CodeEditor: ({ content, readOnly }: { content: string; readOnly?: boolean }) => (
    <div data-testid="code-editor" data-readonly={readOnly}>
      <pre>{content}</pre>
    </div>
  ),
  ContentType: {
    JSON: 'json',
    JSONL: 'jsonl',
    YAML: 'yaml',
    JAVASCRIPT: 'javascript',
  },
}));

// Mock brand assets icons
vi.mock('lucide-react', async () => {
  return (await import('@nemo/testing/mocks/lucide')).mockLucideReact(await import('react'));
});

// Create a test wrapper with theme provider and toast provider
const createWrapper = (theme: 'light' | 'dark' = 'light') => {
  return ({ children }: { children: React.ReactNode }) => (
    <MockToastProvider>
      <ThemeProvider density="standard" theme={theme}>
        {children}
      </ThemeProvider>
    </MockToastProvider>
  );
};

// Mock job data
const mockJob: GenerateJob = {
  id: 'test-job-id',
  name: 'Test Safe Synthesizer Job',
  description: 'Test job description',
  project: 'test-project',
  workspace: 'test-namespace',
  created_at: '2024-01-01T00:00:00.000Z',
  updated_at: '2024-01-01T00:00:00.000Z',
  status: HelixJobStatus.completed,
  spec: {
    data_source: 'hf://test/dataset',
    config: {
      training: {
        batch_size: 8,
        learning_rate: 0.001,
      },
      generation: {
        temperature: 0.8,
        num_records: 1024,
      },
    },
  },
};

describe('JobConfigDrawer', () => {
  const mockOnOpenChange = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('should render the drawer when open is true', () => {
      render(<JobConfigDrawer job={mockJob} open onOpenChange={mockOnOpenChange} />, {
        wrapper: createWrapper(),
      });

      expect(screen.getByText('Job Configuration')).toBeInTheDocument();
      expect(screen.getByTestId('cog-icon')).toBeInTheDocument();
    });
  });

  describe('Drawer Interaction', () => {
    it('should call onOpenChange with false when drawer is closed via SidePanel', async () => {
      render(<JobConfigDrawer job={mockJob} open onOpenChange={mockOnOpenChange} />, {
        wrapper: createWrapper(),
      });

      // Find the close button (SidePanel should render one)
      const closeButton = screen.getByRole('button', { name: /close/i });
      const user = userEvent.setup();
      await user.click(closeButton);

      await waitFor(() => expect(mockOnOpenChange).toHaveBeenCalledWith(false));
    });

    describe('Heading Content', () => {
      it('should render heading with icon and text', () => {
        render(<JobConfigDrawer job={mockJob} open onOpenChange={mockOnOpenChange} />, {
          wrapper: createWrapper(),
        });

        expect(screen.getByText('Job Configuration')).toBeInTheDocument();
        expect(screen.getByTestId('cog-icon')).toBeInTheDocument();
      });
    });
  });
});
