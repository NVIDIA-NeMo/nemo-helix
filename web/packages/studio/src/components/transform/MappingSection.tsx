// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner, Flex, Label, Spinner, Stack } from '@nvidia/foundations-react-core';
import { CustomTemplateRows } from '@studio/components/transform/CustomTemplateRows';
import { FieldMappingRow } from '@studio/components/transform/FieldMappingRow';
import { TemplateSyntaxTooltip } from '@studio/components/transform/TemplateSyntaxTooltip';
import type { TransformMapping } from '@studio/components/transform/useTransformMapping';
import type { FC } from 'react';

interface Props {
  mapping: TransformMapping;
  /** True while the source file is still being read for its columns. */
  isLoadingColumns?: boolean;
}

/**
 * The field mapping itself: one row per field of the chosen format, or the raw
 * key/template grid behind the custom format.
 */
export const MappingSection: FC<Props> = ({ mapping, isLoadingColumns }) => {
  if (isLoadingColumns) {
    return (
      <Flex justify="center" align="center" className="py-[64px]">
        <Spinner slotDescription="Reading source columns..." />
      </Flex>
    );
  }

  return (
    <Stack gap="density-lg">
      <Flex align="center" gap="density-sm">
        <Label className="font-bold">Field mapping</Label>
        <TemplateSyntaxTooltip />
      </Flex>

      {mapping.columns.length === 0 && (
        <Banner kind="inline" status="warning" title="No source columns found">
          The selected file could not be read, so columns cannot be suggested. You can still write
          templates by hand.
        </Banner>
      )}

      {mapping.isCustom ? (
        <CustomTemplateRows
          rows={mapping.customRows}
          columns={mapping.columns}
          onChange={mapping.setCustomRows}
        />
      ) : (
        <Stack gap="density-lg">
          {mapping.format.fields.map((field) => (
            <FieldMappingRow
              key={field.path}
              field={field}
              value={mapping.mappings[field.path] ?? ''}
              columns={mapping.columns}
              isRaw={mapping.rawPaths.has(field.path)}
              generatedIdColumn={mapping.generatedIdColumn}
              onChange={mapping.setMapping}
              onToggleRaw={mapping.toggleRaw}
            />
          ))}
        </Stack>
      )}

      {mapping.missingRequired.length > 0 && (
        <Banner kind="inline" status="warning" title="Required fields are unmapped">
          {mapping.missingRequired.join(', ')} must have a source before this transform can run.
        </Banner>
      )}
    </Stack>
  );
};
