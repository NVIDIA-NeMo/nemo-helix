// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelWorkspaceGroup } from '@nemo/common/src/api/models/useModels';
import { ModelDropdownItem } from '@nemo/common/src/components/ModelSelectV2/ModelDropdownItem';
import type { ModelSelection } from '@nemo/common/src/components/ModelSelectV2/types';
import { VirtualDropdownList } from '@nemo/common/src/components/VirtualDropdownList';
import { creatorToIcon } from '@nemo/common/src/constants/modelMetadata';
import { getURNFromNamedEntityRef } from '@nemo/common/src/namedEntity';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { DropdownHeading, Flex, Text } from '@nvidia/foundations-react-core';
import { useMemo, type FC } from 'react';

type ModelRow =
  | { kind: 'heading'; key: string; workspace: string }
  | { kind: 'model'; key: string; model: ModelEntity; urn: string };

const HEADING_HEIGHT = 32;
const ITEM_HEIGHT = 36;

const estimateSize = (row: ModelRow) => (row.kind === 'heading' ? HEADING_HEIGHT : ITEM_HEIGHT);

export interface ModelDropdownListProps {
  groups: ModelWorkspaceGroup[];
  value: ModelSelection | null;
  onSelect: (selection: ModelSelection) => void;
  hideAdapters?: boolean;
  loading?: boolean;
  /** Called as the user scrolls near the end; no-op when {@link hasMore} is false. */
  onLoadMore?: () => void | Promise<void>;
  hasMore?: boolean;
  isLoadingMore?: boolean;
  /** Footer copy once every page has loaded. Omit to render nothing. */
  doneLoadingMessage?: string;
  emptyMessage?: string;
}

/** The scrollable body of the model dropdown: workspace headings over their models. */
export const ModelDropdownList: FC<ModelDropdownListProps> = ({
  groups,
  value,
  onSelect,
  hideAdapters = false,
  emptyMessage = 'No models found',
  ...listProps
}) => {
  const rows = useMemo<ModelRow[]>(
    () =>
      groups.flatMap((group) => [
        { kind: 'heading' as const, key: `heading-${group.workspace}`, workspace: group.workspace },
        ...group.models.map((model) => {
          const urn = getURNFromNamedEntityRef(model) ?? `${group.workspace}/${model.name}`;
          return { kind: 'model' as const, key: urn, model, urn };
        }),
      ]),
    [groups]
  );

  const renderRow = (row: ModelRow) =>
    row.kind === 'heading' ? (
      <DropdownHeading>
        <Flex gap="density-sm" align="center">
          {creatorToIcon(row.workspace, { className: 'text-base' })}
          <Text>{row.workspace}</Text>
        </Flex>
      </DropdownHeading>
    ) : (
      <ModelDropdownItem
        model={row.model}
        modelUrn={row.urn}
        isSelected={value?.model === row.urn}
        selectedAdapter={value?.model === row.urn ? value.adapter : undefined}
        onSelect={onSelect}
        hideAdapters={hideAdapters}
      />
    );

  return (
    <VirtualDropdownList
      {...listProps}
      rows={rows}
      renderRow={renderRow}
      estimateSize={estimateSize}
      emptyMessage={emptyMessage}
      loadingMessage="Loading models..."
    />
  );
};
