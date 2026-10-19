/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { BaseModelCard } from '@studio/components/BaseModelCard';
import { render, screen } from '@testing-library/react';

const makeModel = (overrides: Partial<ModelEntity> = {}): ModelEntity =>
  ({
    id: 'm1',
    name: 'my-model',
    workspace: 'meta',
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
    ...overrides,
  }) as ModelEntity;

describe('BaseModelCard', () => {
  it('renders the Fine-Tunable badge iff model.fileset is set', () => {
    const { rerender } = render(
      <BaseModelCard
        model={makeModel({ fileset: 'meta/llama-checkpoint' })}
        isChatAvailable={false}
      />
    );
    expect(screen.getByText('Fine-tunable')).toBeInTheDocument();

    rerender(<BaseModelCard model={makeModel({ fileset: undefined })} isChatAvailable={false} />);
    expect(screen.queryByText('Fine-tunable')).not.toBeInTheDocument();
  });

  it('does not render customization badges when customization badges are hidden', () => {
    render(
      <BaseModelCard
        model={makeModel({ fileset: 'meta/llama-checkpoint' })}
        isChatAvailable
        showCustomizationBadges={false}
      />
    );

    expect(screen.queryByText('Fine-tunable')).not.toBeInTheDocument();
  });

  it('renders the Chat indicator iff isChatAvailable is true', () => {
    const { rerender } = render(<BaseModelCard model={makeModel()} isChatAvailable />);
    expect(screen.getByText('Chat')).toBeInTheDocument();

    rerender(<BaseModelCard model={makeModel()} isChatAvailable={false} />);
    expect(screen.queryByText('Chat')).not.toBeInTheDocument();
  });
});
