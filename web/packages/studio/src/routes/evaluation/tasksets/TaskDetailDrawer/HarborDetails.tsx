// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HarborTaskDefinition } from '@nemo/sdk/generated/evals/schema';
import { Anchor, Text } from '@nvidia/foundations-react-core';
import { configSection, parseFilesRef } from '@studio/routes/evaluation/tasksets/helpers';
import { ConfigGrid } from '@studio/routes/evaluation/tasksets/TaskDetailDrawer/ConfigGrid';
import { DetailSection } from '@studio/routes/evaluation/tasksets/TaskDetailDrawer/DetailSection';
import { getFilesetDetailRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { Link } from 'react-router';

export const HarborDetails: FC<{ spec: HarborTaskDefinition }> = ({ spec }) => {
  const archive = parseFilesRef(spec.source.fileset_ref);
  return (
    <>
      <DetailSection heading="Instruction">
        <Text className="whitespace-pre-wrap">{spec.instruction || 'No instruction.'}</Text>
      </DetailSection>
      <DetailSection heading="Environment">
        <ConfigGrid section={configSection(spec.config, 'environment')} empty="Harbor defaults." />
      </DetailSection>
      <DetailSection heading="Tests">
        <ConfigGrid section={configSection(spec.config, 'verifier')} empty="Harbor defaults." />
        <Text className="text-subtle">
          The test scripts ship in the task archive at{' '}
          <Anchor asChild>
            <Link
              to={getFilesetDetailRoute(archive.workspace, archive.fileset)}
              className="break-all"
            >
              {spec.source.fileset_ref}
            </Link>
          </Anchor>
          .
        </Text>
      </DetailSection>
    </>
  );
};
