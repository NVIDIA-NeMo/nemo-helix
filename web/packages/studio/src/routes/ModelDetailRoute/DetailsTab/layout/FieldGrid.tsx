// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KVPair } from '@nemo/common/src/components/KVPair';
import { Grid } from '@nvidia/foundations-react-core';
import type { DetailField } from '@studio/routes/ModelDetailRoute/DetailsTab/fields';
import type { FC } from 'react';

/** Fields without a value are dropped rather than shown as dashes. */
export const FieldGrid: FC<{ fields: DetailField[] }> = ({ fields }) => (
  <Grid colMinWidth="180px" gap="density-xl">
    {fields
      .filter((field) => field.value != null && field.value !== '')
      .map((field) => (
        <KVPair
          key={field.label}
          orientation="vertical"
          label={field.label}
          value={field.value}
          attributes={{ value: { className: 'break-words min-w-0' } }}
        />
      ))}
  </Grid>
);
