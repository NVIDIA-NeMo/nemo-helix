// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import type { UserEvent } from '@testing-library/user-event';

export const selectAutocompleteOption = async (props: {
  user: UserEvent;
  autocompleteEl: HTMLElement;
  option: string;
}) => {
  const { user, autocompleteEl, option } = props;
  await user.click(autocompleteEl);
  const optionNode = await screen.findByRole('option', { name: option });
  await user.click(optionNode);
};
