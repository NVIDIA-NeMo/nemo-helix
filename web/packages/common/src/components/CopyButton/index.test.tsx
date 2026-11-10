// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CopyButton } from '@nemo/common/src/components/CopyButton';
import { MockToastProvider } from '@nemo/common/src/tests/MockToastProvider';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const textToCopy = 'Text that I want to copy';
const toastSuccessText = 'Successfully copied to clipboard!';
const toastErrorText = 'Error copying text to clipboard';

describe('CopyButton', () => {
  it('Copies text to clipboard on-click of the button', async () => {
    const user = userEvent.setup();

    render(
      <MockToastProvider>
        <CopyButton text={textToCopy} />
      </MockToastProvider>
    );

    const button = await screen.findByRole('button');
    await user.click(button);

    expect(await navigator.clipboard.readText()).toEqual(textToCopy);
    expect(await screen.findByText(toastSuccessText)).toBeInTheDocument();
  });

  it('Renders error toast if text fails to copy to clipboard', async () => {
    vitest.spyOn(navigator.clipboard, 'writeText').mockRejectedValueOnce('Error copying text');
    vitest.spyOn(console, 'error').mockImplementation(() => undefined);

    const user = userEvent.setup();

    render(
      <MockToastProvider>
        <CopyButton text={textToCopy} />
      </MockToastProvider>
    );

    const button = await screen.findByRole('button');
    await user.click(button);

    expect(await screen.findByText(toastErrorText)).toBeInTheDocument();
  });
});
