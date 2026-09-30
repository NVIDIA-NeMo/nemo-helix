// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FileTag } from '@studio/components/FileTag/index';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Use the actual KUI components - they are inlined in the test environment.
// FileTag uses Button, Flex, Tag from @nvidia/foundations-react-core plus lucide-react icons.

describe('FileTag', () => {
  describe('when fileName is provided', () => {
    it('renders the file name', () => {
      render(<FileTag fileName="training.jsonl" />);
      expect(screen.getByText('training.jsonl')).toBeInTheDocument();
    });

    it('does not render a "no file" button when a file name is present', () => {
      render(<FileTag fileName="training.jsonl" />);
      // The missingFileNameChip is not rendered when fileName exists
      expect(screen.queryByText('No file detected')).not.toBeInTheDocument();
    });
  });

  describe('when fileName is absent', () => {
    it('renders a button when onNoFileClick is provided', () => {
      render(<FileTag onNoFileClick={vi.fn()} />);

      // The KUI Button renders as a <button> element
      const button = screen.getByRole('button');
      expect(button).toBeInTheDocument();
    });

    it('calls onNoFileClick when the button is clicked', async () => {
      const user = userEvent.setup();
      const onNoFileClick = vi.fn();
      render(<FileTag onNoFileClick={onNoFileClick} />);

      await user.click(screen.getByRole('button'));

      expect(onNoFileClick).toHaveBeenCalledOnce();
    });

    it('does NOT render a button when onNoFileClick is not provided', () => {
      render(<FileTag />);

      // With no onNoFileClick, the component renders a plain <div>, not a button
      expect(screen.queryByRole('button')).not.toBeInTheDocument();
    });

    it('renders the default noFileText when onNoFileClick is provided', () => {
      render(<FileTag onNoFileClick={vi.fn()} />);
      expect(screen.getByText('No file detected')).toBeInTheDocument();
    });

    it('renders the default noFileText when neither fileName nor onNoFileClick is provided', () => {
      render(<FileTag />);
      expect(screen.getByText('No file detected')).toBeInTheDocument();
    });

    it('renders a custom noFileText', () => {
      render(<FileTag noFileText="Select a file" onNoFileClick={vi.fn()} />);
      expect(screen.getByText('Select a file')).toBeInTheDocument();
    });

    it('renders required styling when required=true and no file is present', () => {
      render(<FileTag required />);
      // In required mode, missingFileNameChip uses a <p> with danger styling
      const p = document.querySelector('p');
      expect(p).toBeInTheDocument();
      expect(p).toHaveClass('text-feedback-danger');
    });
  });

  describe('status icons', () => {
    it('renders the success icon when status="success"', () => {
      render(<FileTag fileName="file.jsonl" status="success" />);
      // CircleCheck renders with class lucide-circle-check
      // eslint-disable-next-line testing-library/no-node-access
      expect(document.querySelector('.lucide-circle-check')).toBeInTheDocument();
    });

    it('renders the pending icon when status="pending"', () => {
      render(<FileTag fileName="file.jsonl" status="pending" />);
      // RefreshCw renders with class lucide-refresh-cw
      // eslint-disable-next-line testing-library/no-node-access
      expect(document.querySelector('.lucide-refresh-cw')).toBeInTheDocument();
    });

    it('renders the error icon when status="error"', () => {
      render(<FileTag fileName="file.jsonl" status="error" />);
      // CircleAlert renders with class lucide-circle-alert
      // eslint-disable-next-line testing-library/no-node-access
      expect(document.querySelector('.lucide-circle-alert')).toBeInTheDocument();
    });

    it('renders no status icon when status is absent', () => {
      render(<FileTag fileName="file.jsonl" />);
      // eslint-disable-next-line testing-library/no-node-access
      expect(document.querySelector('.lucide-circle-check')).not.toBeInTheDocument();
      // eslint-disable-next-line testing-library/no-node-access
      expect(document.querySelector('.lucide-refresh-cw')).not.toBeInTheDocument();
      // eslint-disable-next-line testing-library/no-node-access
      expect(document.querySelector('.lucide-circle-alert')).not.toBeInTheDocument();
    });
  });

  describe('regression: replaced div[role=button] with Button component', () => {
    it('clicking element with onNoFileClick is a native button, not a div', () => {
      render(<FileTag onNoFileClick={vi.fn()} />);
      const button = screen.getByRole('button');
      expect(button.tagName).toBe('BUTTON');
    });

    it('without onNoFileClick, the no-file element is not a button', () => {
      const { container } = render(<FileTag />);
      // Should be a plain div, not button
      // eslint-disable-next-line testing-library/no-node-access,testing-library/no-container
      const divs = container.querySelectorAll('div');
      expect(divs.length).toBeGreaterThan(0);
      expect(screen.queryByRole('button')).not.toBeInTheDocument();
    });
  });
});