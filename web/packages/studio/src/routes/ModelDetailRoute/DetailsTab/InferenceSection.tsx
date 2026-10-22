// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { DetailSection } from '@studio/routes/ModelDetailRoute/DetailsTab/layout/DetailSection';
import { FieldGrid } from '@studio/routes/ModelDetailRoute/DetailsTab/layout/FieldGrid';
import type { FC } from 'react';

export const InferenceSection: FC<{ model: ModelEntity }> = ({ model }) => {
  const { url, model_id: modelId } = model.api_endpoint ?? {};
  return (
    <DetailSection
      heading="Inference"
      description="The external endpoint this model is served from."
    >
      <FieldGrid
        fields={[
          { label: 'API Endpoint', value: url },
          { label: 'Model ID', value: modelId },
        ]}
      />
    </DetailSection>
  );
};
