// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DownloadMenu } from '@nemo/common/src/components/DownloadMenu';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { createToastMock } from '@nemo/common/src/providers/toast/useToastMock';
import { downloadPayload } from '@nemo/common/src/utils/download';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

vi.mock('@nemo/common/src/providers/toast/useToast');
vi.mock('@nemo/common/src/utils/download', () => ({ downloadPayload: vi.fn() }));

const payload = { json: { a: 1 }, jsonl: [{ a: 1 }], csv: [{ a: 1 }] };

let toast = createToastMock();
beforeEach(() => {
  toast = createToastMock();
  vi.mocked(useToast).mockReturnValue(toast);
  vi.mocked(downloadPayload).mockReset().mockReturnValue(true);
});

const openMenu = async () => {
  const user = userEvent.setup();
  await user.click(screen.getByTestId('download-menu-trigger'));
  return user;
};

describe('DownloadMenu', () => {
  it('describes what each format contains, with per-format overrides', async () => {
    render(
      <DownloadMenu
        fileName="run"
        getData={() => payload}
        descriptions={{ csv: 'Training steps as a table' }}
      />
    );
    await openMenu();

    expect(screen.getByText('Full record with details and overall scores')).toBeInTheDocument();
    expect(screen.getByText('One result per line, for scripts and datasets')).toBeInTheDocument();
    expect(screen.getByText('Training steps as a table')).toBeInTheDocument();
  });

  it('downloads the resolved data in the chosen format', async () => {
    render(<DownloadMenu fileName="run" getData={async () => payload} />);
    const user = await openMenu();
    await user.click(screen.getByText('JSONL'));

    expect(downloadPayload).toHaveBeenCalledWith(payload, 'run', 'jsonl');
  });

  it('shows a toast when the data cannot be loaded', async () => {
    render(<DownloadMenu fileName="run" getData={() => Promise.reject(new Error('boom'))} />);
    const user = await openMenu();
    await user.click(screen.getByText('CSV'));

    expect(toast.error).toHaveBeenCalledWith('Failed to download: boom');
    expect(downloadPayload).not.toHaveBeenCalled();
  });

  it('warns instead of downloading when there are no results', async () => {
    vi.mocked(downloadPayload).mockReturnValue(false);
    render(<DownloadMenu fileName="run" getData={() => payload} />);
    const user = await openMenu();
    await user.click(screen.getByText('JSON'));

    expect(toast.warning).toHaveBeenCalledWith('There are no results to download.');
  });
});
