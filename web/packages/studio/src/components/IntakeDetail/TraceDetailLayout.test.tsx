// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TraceDetailLayout } from '@studio/components/IntakeDetail/TraceDetailLayout';
import { render, screen } from '@studio/tests/util/render';
import { fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

describe('TraceDetailLayout', () => {
  it('resizes the trajectory sidebar without allowing it below its original width', async () => {
    const user = userEvent.setup();
    render(<TraceDetailLayout navigation="Trajectory">Details</TraceDetailLayout>);

    const getBoundingClientRect = vi
      .spyOn(HTMLElement.prototype, 'getBoundingClientRect')
      .mockReturnValue({
        bottom: 600,
        height: 600,
        left: 0,
        right: 1000,
        top: 0,
        width: 1000,
        x: 0,
        y: 0,
        toJSON: () => ({}),
      });
    vi.spyOn(HTMLElement.prototype, 'offsetWidth', 'get').mockReturnValue(16);

    const resizeHandle = screen.getByRole('separator', {
      name: 'Resize trace trajectory sidebar',
    });
    fireEvent.mouseDown(resizeHandle);
    fireEvent.mouseMove(window, { clientX: 500 });
    fireEvent.mouseMove(window, { clientX: 520 });
    fireEvent.mouseUp(window);

    expect(resizeHandle).toHaveAttribute('aria-valuenow', '520');
    expect(resizeHandle).toHaveAttribute('aria-valuemax', '664');
    expect(getBoundingClientRect).toHaveBeenCalledOnce();

    resizeHandle.focus();
    await user.keyboard('{End}');
    expect(resizeHandle).toHaveAttribute('aria-valuenow', '664');

    await user.keyboard('{Home}');
    expect(resizeHandle).toHaveAttribute('aria-valuenow', '288');
    await user.keyboard('{ArrowRight}');
    expect(resizeHandle).toHaveAttribute('aria-valuenow', '304');
    await user.keyboard('{ArrowLeft}');
    expect(resizeHandle).toHaveAttribute('aria-valuenow', '288');

    fireEvent.mouseDown(resizeHandle);
    fireEvent.mouseMove(window, { clientX: 100 });
    fireEvent.mouseUp(window);
    expect(resizeHandle).toHaveAttribute('aria-valuenow', '288');
  });
});
