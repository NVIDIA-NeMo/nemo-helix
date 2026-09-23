// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KVPair } from '@nemo/common/src/components/KVPair';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { HelixJobTerminalStatuses } from '@nemo/common/src/constants/query';
import { useLiveSeconds } from '@nemo/common/src/hooks/useLiveSeconds';
import {
  formatTimeInSeconds,
  getDifferenceInMilliseconds,
  utcToLocalDate,
} from '@nemo/common/src/utils/date';
import type { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import type { GenerateJob } from '@nemo/sdk/generated/safe-synthesizer/schema';
import { Banner, Button, Divider, Flex, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import { SafeSynthesizerFilesetPreview } from '@studio/components/SafeSynthesizerFilesetPreview';
import { EMPTY_FIELD_VALUE } from '@studio/constants/constants';
import { JobConfigDrawer } from '@studio/routes/SafeSynthesizerJobDetailsRoute/components/JobConfigDrawer';
import { Cog, Play } from 'lucide-react';
import { FC, useState } from 'react';

interface JobDetailsPanelProps {
  job: GenerateJob;
  errorMessage?: string;
}

export const JobDetailsPanel: FC<JobDetailsPanelProps> = ({ job, errorMessage }) => {
  const { id, created_at, status, ownership, updated_at } = job;
  const createdBy = ownership?.created_by;
  const [showJobConfig, setShowJobConfig] = useState(false);
  const isTerminalStatus = HelixJobTerminalStatuses.includes(status as HelixJobStatus);

  const differenceInMilliseconds = getDifferenceInMilliseconds(created_at, updated_at);
  const elapsedSeconds = differenceInMilliseconds
    ? Math.floor(differenceInMilliseconds / 1000)
    : undefined;
  const liveSeconds = useLiveSeconds({
    startDate: !isTerminalStatus ? utcToLocalDate(created_at) : undefined,
  });

  return (
    <>
      <JobConfigDrawer job={job} open={showJobConfig} onOpenChange={setShowJobConfig} />
      <Panel slotHeading="Job Details" slotIcon={<Play />} elevation="high" density="compact">
        <Stack gap="density-xl">
          <Flex gap="density-md" align="center">
            <KVPair
              label="Status"
              value={
                status ? (
                  <Flex align="center" gap="density-sm">
                    <StatusBadge status={status} />
                    {formatTimeInSeconds(
                      HelixJobTerminalStatuses.includes(status as HelixJobStatus)
                        ? elapsedSeconds
                        : liveSeconds
                    )}
                  </Flex>
                ) : (
                  EMPTY_FIELD_VALUE
                )
              }
            />
          </Flex>

          {errorMessage && (
            <Banner
              kind="inline"
              status="error"
              title="Error"
              data-testid="job-details-error-message"
            >
              {errorMessage}
            </Banner>
          )}

          <KVPair
            value={
              id ? (
                <Text kind="label/semibold/md" className="text-left">
                  {id}
                </Text>
              ) : (
                EMPTY_FIELD_VALUE
              )
            }
            label="Job ID"
          />

          <KVPair value={job.name || EMPTY_FIELD_VALUE} label="Name" />
          <KVPair value={job.description || EMPTY_FIELD_VALUE} label="Description" />
          <KVPair
            value={(created_at && <RelativeTime datetime={created_at} />) || EMPTY_FIELD_VALUE}
            label="Created"
          />

          {!!createdBy && <KVPair value={String(createdBy)} label="Created by" />}
          <Divider />
          <SafeSynthesizerFilesetPreview job={job} showJobId={false} />
          <Button
            onClick={() => setShowJobConfig(true)}
            kind="tertiary"
            size="small"
            className="-ml-density-lg"
          >
            <Cog />
            View Job Config
          </Button>
        </Stack>
      </Panel>
    </>
  );
};
