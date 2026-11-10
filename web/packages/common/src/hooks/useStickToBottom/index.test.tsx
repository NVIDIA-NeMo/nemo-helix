// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useStickToBottom } from '@nemo/common/src/hooks/useStickToBottom';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { FC } from 'react';

// jsdom has no layout: scrollHeight/clientHeight report 0 and scrollTop is a no-op.
// Fake the geometry and back scrollTop with a real stored value so assignments stick.
const SCROLL_HEIGHT = 1000;
const CLIENT_HEIGHT = 100;
const MAX_SCROLL_TOP = SCROLL_HEIGHT - CLIENT_HEIGHT;

const mockGeometry = (element: HTMLElement) => {
  let scrollTop = 0;
  Object.defineProperty(element, 'scrollHeight', { configurable: true, value: SCROLL_HEIGHT });
  Object.defineProperty(element, 'clientHeight', { configurable: true, value: CLIENT_HEIGHT });
  Object.defineProperty(element, 'scrollTop', {
    configurable: true,
    get: () => scrollTop,
    set: (value: number) => {
      scrollTop = value;
    },
  });
};

const Harness: FC<{ enabled?: boolean; attached?: boolean }> = ({ enabled, attached = true }) => {
  const { ref, scrollToBottom } = useStickToBottom<HTMLDivElement>({ enabled });
  return (
    <>
      <button onClick={scrollToBottom}>scroll</button>
      {attached && <div ref={ref} data-testid="scroll" />}
    </>
  );
};

it('scrollToBottom() jumps the container to the bottom', async () => {
  const user = userEvent.setup();
  render(<Harness enabled />);
  const scroll = screen.getByTestId('scroll');
  mockGeometry(scroll);
  expect(scroll.scrollTop).toBe(0);

  await user.click(screen.getByRole('button', { name: 'scroll' }));

  expect(scroll.scrollTop).toBe(MAX_SCROLL_TOP);
});

it('scrollToBottom() does not throw before the element is attached', async () => {
  const user = userEvent.setup();
  render(<Harness enabled attached={false} />);

  await expect(user.click(screen.getByRole('button', { name: 'scroll' }))).resolves.not.toThrow();
});
