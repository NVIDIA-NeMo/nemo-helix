// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FormFieldProps } from '@nvidia/foundations-react-core';
import type { UseControllerProps } from 'react-hook-form';

export interface UseControllerComponentProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  useControllerProps: UseControllerProps<any>;
  formFieldProps?: FormFieldProps;
}
