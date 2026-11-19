// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SliderWithTextInput } from '@nemo/common/src/components/SliderWithTextInput';
import type { UseControllerComponentProps } from '@nemo/common/src/types';
import type { ComponentProps } from 'react';
import { useController } from 'react-hook-form';

interface Props
  extends Omit<ComponentProps<typeof SliderWithTextInput>, 'field'>, UseControllerComponentProps {}

export const ControlledSliderWithTextInput = ({ useControllerProps, ...props }: Props) => {
  const { field } = useController(useControllerProps);

  return <SliderWithTextInput field={field} {...props} />;
};
