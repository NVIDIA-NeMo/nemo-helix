// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TextInputSpinner } from '@nemo/common/src/components/form/TextInputSpinner';
import { useModelsListModels as useListModels } from '@nemo/sdk/generated/platform/models';
import {
  Button,
  ComboboxRoot,
  ComboboxTrigger,
  ComboboxContent,
  ComboboxListbox,
  ComboboxItem,
} from '@nvidia/foundations-react-core';
import { keepPreviousData } from '@tanstack/react-query';
import { X } from 'lucide-react';
import { useState, type Dispatch, type SetStateAction } from 'react';
import { useDebounce } from 'use-debounce';

type Props = {
  workspace: string;
  selectedModels: string[];
  setSelectedModels: Dispatch<SetStateAction<string[]>>;
};

export const SearchBaseModels = ({ workspace, selectedModels, setSelectedModels }: Props) => {
  const [filter, setFilter] = useState<string>('');

  const [debouncedBaseModelSearch] = useDebounce(filter, 400);

  const handleChange = (value: string) => setSelectedModels([...selectedModels, value]);

  const { data: models, isFetching } = useListModels(
    workspace,
    {
      page: 1,
      page_size: 20,
      filter: {
        workspace,
      },
    },
    {
      query: {
        enabled: !!debouncedBaseModelSearch,
        placeholderData: keepPreviousData,
      },
    }
  );

  const foundModels =
    models?.data
      ?.filter(
        (model) =>
          model?.base_model &&
          typeof model.base_model === 'string' &&
          !selectedModels.includes(model.base_model)
      )
      .reduce((unique: typeof models.data, model) => {
        if (!unique.some((item) => item?.base_model === model.base_model)) {
          unique.push(model);
        }
        return unique;
      }, []) || [];

  return (
    <>
      <ComboboxRoot
        inputValue={filter}
        onInputValueChange={setFilter}
        onValueChange={(value) => {
          setFilter('');
          if (typeof value === 'string') handleChange(value);
        }}
      >
        <ComboboxTrigger
          placeholder="Search base models..."
          slotEnd={isFetching && <TextInputSpinner />}
        />
        <ComboboxContent>
          <ComboboxListbox
            emptyStateMessage={
              !debouncedBaseModelSearch
                ? 'Start typing to search...'
                : selectedModels.length > 0 && foundModels.length === 0
                  ? 'All available models are already selected'
                  : 'No models found matching your search'
            }
          >
            {foundModels.map((model) => (
              <ComboboxItem
                value={model.base_model as string}
                key={`${model.base_model as string}_option`}
              >
                {model.base_model as string}
              </ComboboxItem>
            ))}
          </ComboboxListbox>
        </ComboboxContent>
      </ComboboxRoot>

      <ul className="w-full mt-2 px-1">
        {selectedModels.map((model) => (
          <li key={`${model}_selected`} className="w-full mt-2 inline-flex justify-between">
            {model}{' '}
            <Button
              aria-label="Remove base model from filter"
              size="tiny"
              kind="tertiary"
              onClick={() =>
                setSelectedModels(selectedModels.filter((selectedModel) => selectedModel !== model))
              }
            >
              <X />
            </Button>
          </li>
        ))}
      </ul>
    </>
  );
};
