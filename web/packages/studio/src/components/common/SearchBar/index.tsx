// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { handleFormErrorsGeneric } from '@nemo/common/src/utils/forms/error';
import { FormField, TextInput } from '@nvidia/foundations-react-core';
import { MagnifyingGlass } from '@nvidia/react-gui-icons/icons';
import { ComponentProps } from 'react';
import { useForm, type SubmitHandler } from 'react-hook-form';

interface FormValues {
  searchString: string;
}

interface Props extends Omit<ComponentProps<typeof TextInput>, 'onSubmit' | 'label'> {
  formClassName?: string;
  onSubmit?: (searchValue: string) => void;
  resetOnSubmit?: boolean;
  label?: string;
}

/**
 * A form wrapped text input for handling search submissions.
 */
export const SearchBar = ({
  formClassName,
  resetOnSubmit,
  onSubmit,
  label,
  ...inputProps
}: Props) => {
  const { register, handleSubmit, setValue } = useForm<FormValues>({
    defaultValues: { searchString: String(inputProps.defaultValue ?? '') },
  });

  const handleOnSubmit: SubmitHandler<FormValues> = (formData) => {
    onSubmit?.(formData.searchString);
    if (resetOnSubmit) {
      setValue('searchString', '');
    }
  };

  return (
    <form
      onSubmit={handleSubmit(
        handleOnSubmit,
        handleFormErrorsGeneric({ title: 'MagnifyingGlass Bar Form Errors' })
      )}
      className={`w-full${formClassName ? ` ${formClassName}` : ''}`}
    >
      <FormField slotLabel={label}>
        <TextInput
          slotStart={<MagnifyingGlass />}
          {...register('searchString')}
          {...inputProps}
          size="small"
        />
      </FormField>
    </form>
  );
};
