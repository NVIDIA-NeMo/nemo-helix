// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  DeleteConfirmationModal,
  type DeleteModalProps,
} from '@nemo/common/src/components/DeleteConfirmationModal';
import { render, screen } from '@studio/tests/util/render';
import { act, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AxiosError, AxiosHeaders } from 'axios';
import { useState } from 'react';

const DESCRIPTION = 'Deletes the model weights and stops any deployments that serve them.';

const renderModal = (props: Partial<DeleteModalProps> = {}) => {
  const onClose = vi.fn();
  const onNotify = vi.fn();
  const onDelete = vi.fn().mockResolvedValue(true);
  render(
    <DeleteConfirmationModal
      open
      onClose={onClose}
      onDelete={onDelete}
      onNotify={onNotify}
      entity="model"
      ids={['llama-7b']}
      description={DESCRIPTION}
      {...props}
    />
  );
  return { onClose, onNotify, onDelete, user: userEvent.setup() };
};

describe('DeleteConfirmationModal', () => {
  it('names the entity and item in the heading and button', async () => {
    renderModal();

    const dialog = await screen.findByRole('dialog');
    expect(within(dialog).getByText("Delete model 'llama-7b'?")).toBeInTheDocument();
    expect(within(dialog).getByText(DESCRIPTION)).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: 'Delete Model' })).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
    expect(within(dialog).queryByRole('list')).not.toBeInTheDocument();
    expect(within(dialog).queryByRole('textbox')).not.toBeInTheDocument();
  });

  it('marks Cancel as the initially focused button', async () => {
    renderModal();

    expect(await screen.findByRole('button', { name: 'Cancel' })).toHaveAttribute('autofocus');
    expect(screen.getByRole('button', { name: 'Delete Model' })).not.toHaveAttribute('autofocus');
  });

  it('uses the count and lists the first three ids for several items', async () => {
    renderModal({ ids: ['alpha', 'beta', 'gamma', 'delta', 'epsilon'] });

    expect(await screen.findByText('Delete 5 models?')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Delete 5 Models' })).toBeInTheDocument();
    const list = screen.getByRole('list');
    expect(within(list).getAllByRole('listitem')).toHaveLength(3);
    ['alpha', 'beta', 'gamma'].forEach((name) => {
      expect(within(list).getByText(name)).toBeInTheDocument();
    });
    expect(screen.queryByText('delta')).not.toBeInTheDocument();
    expect(screen.getByText('and 2 more')).toBeInTheDocument();
  });

  it('omits the overflow line when every name fits', async () => {
    renderModal({ ids: ['a', 'b'] });

    expect(within(await screen.findByRole('list')).getAllByRole('listitem')).toHaveLength(2);
    expect(screen.queryByText(/more$/)).not.toBeInTheDocument();
  });

  it('uses entityPlural when given', async () => {
    renderModal({ entity: 'policy', entityPlural: 'policies', ids: ['a', 'b'] });

    expect(await screen.findByText('Delete 2 policies?')).toBeInTheDocument();
  });

  it('switches the copy for the remove verb', async () => {
    const { onNotify, user } = renderModal({ entity: 'member', ids: ['alice'], verb: 'remove' });

    expect(screen.getByText("Remove member 'alice'?")).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Remove Member' }));

    await waitFor(() =>
      expect(onNotify).toHaveBeenCalledWith("Removed member 'alice'.", 'success', undefined)
    );
  });

  it('deletes, closes and shows the generated success toast', async () => {
    const { onDelete, onClose, onNotify, user } = renderModal();

    await user.click(screen.getByRole('button', { name: 'Delete Model' }));

    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(onDelete).toHaveBeenCalledTimes(1);
    expect(onNotify).toHaveBeenCalledWith("Deleted model 'llama-7b'.", 'success', undefined);
  });

  it('shows the generated error toast when the delete reports failure', async () => {
    const { onNotify, user } = renderModal({
      ids: ['a', 'b'],
      onDelete: vi.fn().mockResolvedValue(false),
    });

    await user.click(screen.getByRole('button', { name: 'Delete 2 Models' }));

    await waitFor(() =>
      expect(onNotify).toHaveBeenCalledWith("Couldn't delete 2 models.", 'error', undefined)
    );
  });

  it('shows the API detail when the delete throws', async () => {
    const error = new AxiosError('Request failed with status code 409', '409', undefined, null, {
      status: 409,
      statusText: 'Conflict',
      headers: {},
      config: { headers: new AxiosHeaders() },
      data: { detail: 'Model is still deployed.' },
    });
    const { onNotify, user } = renderModal({ onDelete: vi.fn().mockRejectedValue(error) });

    await user.click(screen.getByRole('button', { name: 'Delete Model' }));

    await waitFor(() =>
      expect(onNotify).toHaveBeenCalledWith('Model is still deployed.', 'error', undefined)
    );
  });

  it('prefers caller-supplied toast text', async () => {
    const { onNotify, user } = renderModal({ successText: 'Deletion started.' });

    await user.click(screen.getByRole('button', { name: 'Delete Model' }));

    await waitFor(() =>
      expect(onNotify).toHaveBeenCalledWith('Deletion started.', 'success', undefined)
    );
  });

  it('keeps Cancel enabled while the delete is pending', async () => {
    const onDelete = vi.fn(() => new Promise<boolean>(() => {}));
    const { onClose, user } = renderModal({ onDelete });

    const submit = screen.getByRole('button', { name: 'Delete Model' });
    await user.click(submit);

    await waitFor(() => expect(submit).toBeDisabled());
    const cancel = screen.getByRole('button', { name: 'Cancel' });
    expect(cancel).toBeEnabled();
    await user.click(cancel);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('counts the target in the heading and button when ids is empty', async () => {
    renderModal({ ids: [] });

    expect(await screen.findByText('Delete 0 models?')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Delete 0 Models' })).toBeInTheDocument();
  });

  it('uses the verb in the fallback success toast when no entity is given', async () => {
    const { onNotify, user } = renderModal({
      entity: undefined,
      ids: undefined,
      title: 'Remove member',
      verb: 'remove',
    });

    await user.click(screen.getByRole('button', { name: 'Remove' }));

    await waitFor(() =>
      expect(onNotify).toHaveBeenCalledWith('Successfully removed!', 'success', undefined)
    );
  });

  it('does not close a reopened dialog when an abandoned delete settles', async () => {
    let resolveDelete!: (ok: boolean) => void;
    const onDelete = vi.fn(
      () =>
        new Promise<boolean>((resolve) => {
          resolveDelete = resolve;
        })
    );
    const onNotify = vi.fn();
    const closes: string[] = [];

    const Harness = () => {
      const [open, setOpen] = useState(true);
      return (
        <>
          <button type="button" onClick={() => setOpen(true)}>
            Reopen
          </button>
          <DeleteConfirmationModal
            open={open}
            onClose={() => {
              closes.push('close');
              setOpen(false);
            }}
            onDelete={onDelete}
            onNotify={onNotify}
            entity="model"
            ids={['llama-7b']}
            description={DESCRIPTION}
          />
        </>
      );
    };

    render(<Harness />);
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'Delete Model' }));
    await waitFor(() => expect(onDelete).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(closes).toHaveLength(1);

    await user.click(screen.getByRole('button', { name: 'Reopen', hidden: true }));
    const submit = await screen.findByRole('button', { name: 'Delete Model' });
    expect(submit).toBeEnabled();

    await act(async () => {
      resolveDelete(true);
    });

    expect(onNotify).toHaveBeenCalledWith("Deleted model 'llama-7b'.", 'success', undefined);
    expect(closes).toHaveLength(1);
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Delete Model' })).toBeEnabled();
  });
});
