// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TextInputSpinner } from '@nemo/common/src/components/form/TextInputSpinner';
import { creatorToIcon } from '@nemo/common/src/constants/modelMetadata';
import { Filter } from '@nemo/icons/icons';
import {
  Badge,
  Block,
  DropdownHeading,
  DropdownSection,
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
import { useAutofillFromSearchParams } from '@studio/hooks/evaluation/useAutofillFromSearchParams';
import {
  type EvaluationModelItem,
  useEvaluationModels,
} from '@studio/hooks/evaluation/useEvaluationModels';
import { useSetFieldErrorOnApiError } from '@studio/hooks/evaluation/useSetFieldErrorOnApiError';
import { type ChangeEvent, useCallback, useMemo, useRef, useState } from 'react';
import { type FieldValues, type Path, useController, useFormContext } from 'react-hook-form';

export interface EvaluationModelSelectProps<TFieldValues extends FieldValues = FieldValues> {
  required?: boolean;
  placeholder?: string;
  formFieldName: Path<TFieldValues>;
  autofillFromSearchParams?: boolean;
}

export const EvaluationModelSelect = <TFieldValues extends FieldValues = FieldValues>({
  required = false,
  placeholder = 'Select a model to evaluate',
  formFieldName,
  autofillFromSearchParams = true,
}: EvaluationModelSelectProps<TFieldValues>) => {
  const {
    control,
    formState: { disabled, isSubmitting },
  } = useFormContext<TFieldValues>();
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const filterInputRef = useRef<HTMLInputElement>(null);

  const { items, isLoading, error } = useEvaluationModels({
    search,
    enabled: !disabled && open,
  });

  useSetFieldErrorOnApiError<TFieldValues>(formFieldName, error);

  useAutofillFromSearchParams({
    searchParamName: 'model',
    fieldName: formFieldName,
    enabled: autofillFromSearchParams,
  });

  const {
    field: { onBlur: onBlurField, onChange: onChangeField, value, disabled: fieldDisabled },
    fieldState: { error: fieldError },
  } = useController({ name: formFieldName, control });

  const groupedItems = useMemo(() => {
    const groups = new Map<string, EvaluationModelItem[]>();
    for (const item of items) {
      const ws = item.model.workspace;
      if (!groups.has(ws)) groups.set(ws, []);
      groups.get(ws)!.push(item);
    }
    return groups;
  }, [items]);

  const selectedItem = useMemo(() => items.find((item) => item.value === value), [items, value]);

  const handleOpenChange = useCallback((nextOpen: boolean) => {
    setOpen(nextOpen);
    if (!nextOpen) {
      setSearch('');
      return;
    }
    setTimeout(() => filterInputRef.current?.focus(), 0);
  }, []);

  const handleSearchChange = (e: ChangeEvent<HTMLInputElement>) => {
    setSearch(e.target.value);
  };

  const creator = selectedItem?.model.workspace;

  return (
    <FormField
      slotLabel="Model"
      required={required}
      status={fieldError ? 'error' : undefined}
      slotError={fieldError?.message}
    >
      <SelectRoot
        disabled={fieldDisabled || isSubmitting}
        value={value ?? undefined}
        onValueChange={(val: string) => onChangeField(val)}
        onOpenChange={handleOpenChange}
      >
        <SelectTrigger
          className={`w-full border-1 ${fieldDisabled || isSubmitting ? 'nv-input-disabled' : 'nv-input'} relative`}
          onBlur={onBlurField}
          placeholder={isLoading ? 'Loading Models...' : placeholder}
          slotStart={creator && creatorToIcon(creator, { className: 'text-base' })}
          slotEnd={isLoading && <TextInputSpinner />}
          required={required}
          status={fieldError ? 'error' : undefined}
        />
        <SelectContent className="w-(--radix-popper-anchor-width)">
          <SelectListbox>
            <Block className="p-2 w-full sticky top-0 bg-surface z-10">
              <TextInput
                ref={filterInputRef}
                name="model-filter"
                className="overflow-hidden"
                slotStart={<Filter />}
                placeholder="Search..."
                value={search}
                onChange={handleSearchChange}
                attributes={{
                  Input: {
                    ['data-testid' as never]: 'model-filter',
                  },
                }}
              />
            </Block>
            <Stack className="overflow-auto w-full max-h-[300px]">
              {groupedItems.size > 0 ? (
                Array.from(groupedItems.entries()).map(([workspace, wsItems]) => (
                  <DropdownSection key={workspace}>
                    <DropdownHeading>
                      <Flex gap="density-sm" align="center">
                        {creatorToIcon(workspace, { className: 'text-base' })}
                        <Text className="font-bold">{workspace}</Text>
                      </Flex>
                    </DropdownHeading>
                    {wsItems.map((item) => (
                      <SelectItem key={item.value} className="relative" value={item.value}>
                        <Flex className="w-full" align="center" justify="between" gap="density-sm">
                          <Text className="truncate flex-1">
                            {item.adapter
                              ? `${item.model.name} / ${item.adapter.name}`
                              : item.model.name}
                          </Text>
                          {item.adapter && (
                            <Badge kind="solid" color="green" className="shrink-0">
                              LoRA
                            </Badge>
                          )}
                        </Flex>
                      </SelectItem>
                    ))}
                  </DropdownSection>
                ))
              ) : (
                <DropdownSection>
                  {!isLoading && (
                    <DropdownHeading>
                      <Text>No Models Found</Text>
                    </DropdownHeading>
                  )}
                </DropdownSection>
              )}
            </Stack>
          </SelectListbox>
        </SelectContent>
      </SelectRoot>
    </FormField>
  );
};
