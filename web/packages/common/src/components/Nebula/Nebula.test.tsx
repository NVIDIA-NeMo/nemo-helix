// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Nebula } from '@nemo/common/src/components/Nebula/index';
import { render, screen } from '@testing-library/react';

// Mock the animation module – tests are not concerned with canvas drawing logic.
vi.mock('@nemo/common/src/components/Nebula/animate', () => ({
  initialize: vi.fn(),
}));

describe('Nebula', () => {
  describe('canvas accessibility attributes', () => {
    it('sets aria-hidden="true" on the canvas element', () => {
      render(<Nebula />);

      // eslint-disable-next-line testing-library/no-node-access
      const canvas = document.querySelector('canvas');
      expect(canvas).toBeInTheDocument();
      expect(canvas).toHaveAttribute('aria-hidden', 'true');
    });

    it('sets tabIndex={-1} on the canvas to remove it from the tab order', () => {
      render(<Nebula />);

      // eslint-disable-next-line testing-library/no-node-access
      const canvas = document.querySelector('canvas');
      expect(canvas).toBeInTheDocument();
      expect(canvas).toHaveAttribute('tabindex', '-1');
    });

    it('hides the canvas from the accessibility tree (canvas is not in the document by role)', () => {
      render(<Nebula />);

      // aria-hidden removes the element from the accessibility tree, so it has no ARIA role.
      // Verify it cannot be found via an accessible query.
      expect(screen.queryByRole('img')).not.toBeInTheDocument();
    });
  });

  describe('container', () => {
    it('renders the outer container with data-testid="nv-nebula"', () => {
      render(<Nebula />);

      expect(screen.getByTestId('nv-nebula')).toBeInTheDocument();
    });

    it('applies a custom className to the outer container', () => {
      render(<Nebula className="my-custom-class" />);

      expect(screen.getByTestId('nv-nebula')).toHaveClass('my-custom-class');
    });
  });

  describe('canvas element is present (regression: ensures canvas is not removed)', () => {
    it('renders exactly one canvas element', () => {
      render(<Nebula />);

      // eslint-disable-next-line testing-library/no-node-access
      const canvases = document.querySelectorAll('canvas');
      expect(canvases).toHaveLength(1);
    });
  });
});