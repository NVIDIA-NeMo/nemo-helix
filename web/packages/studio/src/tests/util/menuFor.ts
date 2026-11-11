// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';

export const menuFor = (trigger: HTMLElement): HTMLElement => {
  const menu = screen
    .getAllByRole('menu')
    .find(({ id }) => id === trigger.getAttribute('aria-controls'));
  if (!menu) {
    throw new Error('menu for trigger not found');
  }
  return menu;
};
