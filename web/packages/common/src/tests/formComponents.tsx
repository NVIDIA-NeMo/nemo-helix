// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FC, PropsWithChildren } from 'react';
import { FormProvider, useForm, type UseFormProps } from 'react-hook-form';

export const FormWrapper: FC<PropsWithChildren<{ formProps?: UseFormProps }>> = ({
  children,
  formProps,
}) => {
  const methods = useForm(formProps);
  return <FormProvider {...methods}>{children}</FormProvider>;
};
