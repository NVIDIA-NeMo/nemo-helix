// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KVPair } from '@nemo/common/src/components/KVPair';
import type { FilesetOutput, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { Divider, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import { getMetadataSections } from '@studio/routes/FilesetDetailRoute/FilesetMetadataPanel/utils';
import type { FC } from 'react';

export interface FilesetMetadataPanelProps {
  fileset: FilesetOutput;
  readmeMetadata?: Record<string, unknown>;
  modelEntities?: ModelEntity[];
}

export const FilesetMetadataPanel: FC<FilesetMetadataPanelProps> = ({
  fileset,
  readmeMetadata,
  modelEntities = [],
}) => {
  const sections = getMetadataSections(fileset, readmeMetadata, modelEntities);

  return (
    <Panel
      elevation="high"
      density="compact"
      className="w-full"
      data-testid="fileset-metadata-panel"
    >
      <Stack gap="density-xl">
        {sections.map((section, index) => (
          <Stack key={section.value} gap="density-lg">
            {index > 0 ? <Divider /> : null}
            <Text kind="label/bold/sm">{section.title}</Text>
            <Stack gap="density-md">
              {section.rows.map((row) => (
                <KVPair
                  key={`${section.value}-${row.label}`}
                  label={row.label}
                  value={row.value}
                  orientation="horizontal"
                  size="narrow"
                  attributes={{
                    value: { className: 'min-w-0 break-words text-wrap' },
                  }}
                />
              ))}
            </Stack>
          </Stack>
        ))}
      </Stack>
    </Panel>
  );
};
