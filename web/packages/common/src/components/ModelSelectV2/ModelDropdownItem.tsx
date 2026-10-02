// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ModelDetailsPanel } from '@nemo/common/src/components/ModelSelectV2/ModelDetailsPanel';
import type { ModelSelection } from '@nemo/common/src/components/ModelSelectV2/types';
import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import {
  Divider,
  DropdownHeading,
  DropdownItem,
  DropdownSub,
  DropdownSubContent,
  DropdownSubTrigger,
  Flex,
  Text,
} from '@nvidia/foundations-react-core';
import { Check } from 'lucide-react';
import { memo, useState, type FC } from 'react';

interface ModelDropdownItemProps {
  model: ModelEntity;
  /** Precomputed by the list, so the URN is parsed once per model rather than once per render. */
  modelUrn: string;
  /** Whether this model is the current selection. Primitive so the row can memoize. */
  isSelected: boolean;
  /** The selected adapter, when it belongs to this model. */
  selectedAdapter?: string;
  onSelect: (selection: ModelSelection) => void;
  hideAdapters?: boolean;
}

const formatDate = (dateString: string | undefined): string => {
  if (!dateString) return '';
  const date = new Date(dateString);
  return date.toLocaleDateString(undefined, { year: 'numeric', month: '2-digit', day: '2-digit' });
};

const ModelName: FC<{ name: string | undefined }> = ({ name }) => {
  const baseName = name?.split('@')[0];
  const version = name?.includes('@') ? name.split('@')[1] : undefined;

  return (
    <Flex className="w-full" align="center" justify="between">
      <Text className="truncate flex-1">{baseName}</Text>
      {version && (
        <Text
          className="text-secondary truncate ml-2 max-w-16 text-left"
          style={{ direction: 'rtl' }} // eslint-disable-line no-restricted-syntax -- RTL truncation for version suffix
        >
          {version}
        </Text>
      )}
    </Flex>
  );
};

/**
 * Tracks whether a submenu has ever been opened.
 *
 * KUI's `DropdownSubContent` is a native `popover="auto"` surface with no presence gating — it
 * sits in the DOM whether or not the submenu is open. Rendering its body unconditionally means
 * every row in view mounts a whole details panel (two live relative-time tickers each), and every
 * adapter mounts another one below that. Deferring the body until first hover keeps the cost
 * proportional to what the user actually looks at; latching keeps repeat hovers instant.
 */
const useOpenedOnce = () => {
  const [hasOpened, setHasOpened] = useState(false);
  const handleOpenChange = (open: boolean) => {
    if (open) setHasOpened(true);
  };
  return [hasOpened, handleOpenChange] as const;
};

const AdapterItem: FC<{
  adapter: Adapter;
  model: ModelEntity;
  modelUrn: string;
  isSelected: boolean;
  onSelect: (selection: ModelSelection) => void;
}> = ({ adapter, model, modelUrn, isSelected, onSelect }) => {
  const [hasOpened, handleOpenChange] = useOpenedOnce();

  return (
    <DropdownSub onOpenChange={handleOpenChange}>
      <DropdownSubTrigger
        slotEnd={false}
        data-testid="model-dropdown-adapter-option"
        onClick={() => onSelect({ model: modelUrn, adapter: adapter.name, entity: model })}
      >
        <Flex className="w-full" align="center" justify="between" gap="density-md">
          <Flex align="center" gap="density-sm" className="min-w-0">
            {isSelected && (
              <Check size={14} className="shrink-0" data-testid="model-dropdown-selected-check" />
            )}
            <Text className="truncate">{adapter.name}</Text>
          </Flex>
          {adapter.created_at && (
            <Text className="text-secondary whitespace-nowrap" kind="body/regular/sm">
              {formatDate(adapter.created_at)}
            </Text>
          )}
        </Flex>
      </DropdownSubTrigger>
      {/* eslint-disable-next-line no-restricted-syntax -- KUI ignores Tailwind width classes */}
      <DropdownSubContent style={{ width: 360 }}>
        {hasOpened && <ModelDetailsPanel model={model} adapter={adapter} />}
      </DropdownSubContent>
    </DropdownSub>
  );
};

const sortAdaptersByNewest = (adapters: Adapter[]): Adapter[] =>
  [...adapters].sort((a, b) => {
    if (!a.created_at || !b.created_at) return 0;
    return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
  });

const ModelDropdownItemImpl: FC<ModelDropdownItemProps> = ({
  model,
  modelUrn,
  isSelected,
  selectedAdapter,
  onSelect,
  hideAdapters = false,
}) => {
  const [hasOpened, handleOpenChange] = useOpenedOnce();
  const hasAdapters = !hideAdapters && model.adapters && model.adapters.length > 0;

  if (!hasAdapters) {
    return (
      <DropdownSub onOpenChange={handleOpenChange}>
        <DropdownSubTrigger
          slotEnd={false}
          data-testid="model-dropdown-item"
          onClick={() => onSelect({ model: modelUrn, entity: model })}
        >
          <ModelName name={model.name} />
        </DropdownSubTrigger>
        {/* eslint-disable-next-line no-restricted-syntax -- KUI ignores Tailwind width classes */}
        <DropdownSubContent style={{ width: 360 }}>
          {hasOpened && <ModelDetailsPanel model={model} />}
        </DropdownSubContent>
      </DropdownSub>
    );
  }

  const isBaseSelected = isSelected && !selectedAdapter;

  return (
    <DropdownSub onOpenChange={handleOpenChange}>
      <DropdownSubTrigger data-testid="model-dropdown-item-with-adapters">
        <ModelName name={model.name} />
      </DropdownSubTrigger>
      {/* eslint-disable-next-line no-restricted-syntax -- KUI ignores Tailwind width classes */}
      <DropdownSubContent style={{ width: 360 }}>
        {hasOpened && (
          <>
            <DropdownHeading>Base Model</DropdownHeading>
            <DropdownItem
              data-testid="model-dropdown-base-option"
              onSelect={() => onSelect({ model: modelUrn, entity: model })}
            >
              <Flex align="center" gap="density-sm">
                {isBaseSelected && (
                  <Check
                    size={14}
                    className="shrink-0"
                    data-testid="model-dropdown-selected-check"
                  />
                )}
                <Text>{modelUrn}</Text>
              </Flex>
            </DropdownItem>
            <Divider />
            <DropdownHeading>Adapters</DropdownHeading>
            {sortAdaptersByNewest(model.adapters!).map((adapter) => (
              <AdapterItem
                key={adapter.name}
                adapter={adapter}
                model={model}
                modelUrn={modelUrn}
                isSelected={selectedAdapter === adapter.name}
                onSelect={onSelect}
              />
            ))}
          </>
        )}
      </DropdownSubContent>
    </DropdownSub>
  );
};

/**
 * Memoized because the filter box's state lives above the list: without this, every keystroke
 * re-renders every row in view. All props are primitives or stable references, so the shallow
 * compare actually holds.
 */
export const ModelDropdownItem = memo(ModelDropdownItemImpl);
