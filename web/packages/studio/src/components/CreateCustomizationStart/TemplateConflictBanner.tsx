// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { Banner, Button, Flex, Modal, Stack, Text } from '@nvidia/foundations-react-core';
import type { TemplateFilesetConflict } from '@studio/components/CreateCustomizationStart/templatePreflight';
import { getFilesetRoute } from '@studio/routes/utils';
import { type FC, useState } from 'react';
import { Link } from 'react-router';

interface Props {
  conflict: TemplateFilesetConflict;
  /** Build the recipe's data under the next free name, leaving the conflicting fileset alone. */
  onRename: () => void;
  /** Delete the conflicting fileset, then build the recipe's data again. */
  onReplace: () => void;
  /** True while a chosen recovery is running, so the actions cannot be fired twice. */
  busy: boolean;
}

const WHICH_FILESET: Record<TemplateFilesetConflict['target'], string> = {
  source: 'source dataset',
  converted: 'training dataset',
};

/**
 * Turns a fileset name collision into something the user can act on without leaving the
 * page. Every recipe writes to a fixed name, so in a shared workspace the second person to
 * run one lands here; before this existed, the run simply stopped with a sentence telling
 * them to go rename something.
 *
 * The conflicting fileset is a link rather than quoted text — the first question is always
 * "what is already there?", and it is not answerable from the message alone.
 */
export const TemplateConflictBanner: FC<Props> = ({ conflict, onRename, onReplace, busy }) => {
  const [confirmingReplace, setConfirmingReplace] = useState(false);

  return (
    <>
      <Banner kind="inline" status="warning">
        <Stack gap="density-md">
          <Text kind="body/regular/md">
            The recipe&apos;s {WHICH_FILESET[conflict.target]}{' '}
            <Link
              to={getFilesetRoute(conflict.workspace, conflict.filesetRef)}
              className="text-primary underline"
            >
              {conflict.filesetRef}
            </Link>{' '}
            {conflict.message}
          </Text>

          <Flex gap="density-md" wrap="wrap">
            <Button asChild kind="secondary">
              <Link to={getFilesetRoute(conflict.workspace, conflict.filesetRef)}>
                Open fileset
              </Link>
            </Button>
            <LoadingButton kind="secondary" loading={busy} onClick={onRename}>
              Create under a different name
            </LoadingButton>
            <Button
              kind="tertiary"
              color="danger"
              disabled={busy}
              onClick={() => setConfirmingReplace(true)}
            >
              Replace it
            </Button>
          </Flex>
        </Stack>
      </Banner>

      <Modal
        open={confirmingReplace}
        onOpenChange={(next) => !next && setConfirmingReplace(false)}
        slotHeading={`Replace ${conflict.filesetRef}?`}
      >
        <Stack gap="density-xl">
          <Text kind="body/regular/md">
            This permanently deletes the fileset and everything in it, then rebuilds it from the
            recipe. Jobs that already trained on it are not affected, but anything else pointing at
            it will lose its data.
          </Text>

          <Flex justify="end" gap="density-md">
            <Button kind="tertiary" onClick={() => setConfirmingReplace(false)}>
              Cancel
            </Button>
            <LoadingButton
              color="danger"
              loading={busy}
              onClick={() => {
                setConfirmingReplace(false);
                onReplace();
              }}
            >
              Delete and rebuild
            </LoadingButton>
          </Flex>
        </Stack>
      </Modal>
    </>
  );
};
