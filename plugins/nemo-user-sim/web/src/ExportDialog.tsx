// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FormModal } from '@nemo/common';
import {
  Checkbox,
  Flex,
  FormField,
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectRoot,
  SelectTrigger,
  Stack,
  Text,
  TextInput,
} from '@nvidia/foundations-react-core';
import { useState } from 'react';
import { useCreateJob } from './api';
import { useHost } from './host';
import type { ExportRequest } from './types';

// The trigger echoes the raw value, so values are the display labels.
const QUALITY = { Strict: 'strict', Generous: 'generous' } as const;

export const ExportDialog = ({
  jobs,
  open,
  onClose,
}: {
  jobs: string[];
  open: boolean;
  onClose: () => void;
}) => {
  const { notifications } = useHost();
  const create = useCreateJob<ExportRequest>('export');
  const [fileset, setFileset] = useState(
    jobs.length === 1 ? `${jobs[0]}-training` : 'user-sim-training'
  );
  const [quality, setQuality] = useState<keyof typeof QUALITY>('Strict');
  const [requireReview, setRequireReview] = useState(false);

  const submit = () =>
    create.mutate(
      { jobs, fileset: fileset.trim(), quality: QUALITY[quality], require_review: requireReview },
      {
        onSuccess: (exportJob) => {
          onClose();
          notifications.notify(
            `Export ${exportJob.name} started. Training data goes to fileset ${fileset.trim()}.`,
            'success'
          );
        },
        onError: (error) =>
          notifications.notify(`Couldn't start the export. ${error.message}`, 'error'),
      }
    );

  return (
    <FormModal
      open={open}
      title="Export training data"
      submitButtonText="Export"
      loading={create.isPending}
      submitDisabled={!fileset.trim()}
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
      onClose={onClose}
    >
      <Stack gap="density-md">
        <Text kind="body/regular/sm" className="text-subtle">
          Writes the conversations that passed scoring in{' '}
          {jobs.length === 1 ? 'this run' : `these ${jobs.length} runs`} as one chat JSONL dataset,
          ready for the post-training plugin.
        </Text>
        <FormField
          name="fileset"
          slotLabel="Dataset fileset"
          slotHelp="Created if it doesn't exist."
        >
          <TextInput value={fileset} onChange={(e) => setFileset(e.target.value)} />
        </FormField>
        <FormField
          name="quality"
          slotLabel="Quality gate"
          slotHelp="Strict drops any conversation a judge flagged."
        >
          <SelectRoot
            value={quality}
            onValueChange={(v: string) => setQuality(v as keyof typeof QUALITY)}
          >
            <SelectTrigger className="w-full" />
            <SelectContent className="w-(--radix-popper-anchor-width)">
              <SelectListbox>
                {Object.keys(QUALITY).map((label) => (
                  <SelectItem key={label} value={label}>
                    {label}
                  </SelectItem>
                ))}
              </SelectListbox>
            </SelectContent>
          </SelectRoot>
        </FormField>
        <Flex align="center" gap="density-sm">
          <Checkbox
            checked={requireReview}
            onCheckedChange={(checked) => setRequireReview(checked === true)}
            attributes={{ CheckboxInput: { 'aria-label': 'Require review' } }}
          />
          <Text kind="body/regular/sm">Only conversations a reviewer gave a thumbs up</Text>
        </Flex>
      </Stack>
    </FormModal>
  );
};
