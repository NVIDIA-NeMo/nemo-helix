// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Grid } from '@nvidia/foundations-react-core';
import { TemplateCard } from '@studio/components/CreateCustomizationStart/TemplateCard';
import type { TemplateGridProps } from '@studio/components/CreateCustomizationStart/types';
import { CUSTOMIZATION_TEMPLATES } from '@studio/constants/customizationTemplates';
import type { FC } from 'react';

export const TemplateGrid: FC<TemplateGridProps> = ({ selectedTemplateId, onSelectTemplate }) => (
  // Fixed columns rather than an auto-fill min-width, so a short list keeps card-sized cards
  // instead of stretching one across the whole row.
  <Grid cols={{ base: 1, md: 2, lg: 3 }} gap="density-md">
    {CUSTOMIZATION_TEMPLATES.map((template) => (
      <TemplateCard
        key={template.id}
        template={template}
        selected={selectedTemplateId === template.id}
        onSelect={() => onSelectTemplate(template.id)}
      />
    ))}
  </Grid>
);
