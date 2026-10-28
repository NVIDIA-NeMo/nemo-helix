// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { InsightsModelPairFields } from '@studio/components/ImportTracesModal/InsightsModelPairFields';
import { render, screen } from '@testing-library/react';

vi.mock('@nemo/common/src/components/ModelSelectV2', () => ({
  WorkspaceModelSelect: ({ value }: { value: { model: string } | null }) => (
    <span>{value?.model}</span>
  ),
}));

describe.each([
  'default/nvidia-nvidia-nemotron-3-5-lightning-30b',
  'default/nvidia-nemotron-3.5-lightning-30b-a3b',
  'default/nvidia_nemotron_3_5_lightning_30b_custom',
  'default/nvidia_nemotron_3_5_lightning_30b',
  'default/nvidia_nvidia_nemotron_3_5_lightning_30b',
  'default/nvidia-nemotron_3.5-lightning_30b',
])('Compass model guidance for %s', (lightning) => {
  it.each(['default', 'fast'] as const)(
    'warns for the effective %s model and clears on override',
    (role) => {
      const props = {
        workspace: 'default',
        unresolved: false,
        defaultModel: '',
        fastModel: '',
        onDefaultModelChange: vi.fn(),
        onFastModelChange: vi.fn(),
        stored: { [`${role}Model`]: lightning },
      };
      const { rerender } = render(<InsightsModelPairFields {...props} />);
      expect(screen.getByText(/known tool-calling failures/)).toHaveTextContent('Super VL');
      rerender(
        <InsightsModelPairFields
          {...props}
          {...{ [`${role}Model`]: 'default/nemotron-super-vl' }}
        />
      );
      expect(screen.queryByText(/known tool-calling failures/)).not.toBeInTheDocument();
      rerender(
        <InsightsModelPairFields
          {...props}
          {...{ [`${role}Model`]: 'default/nvidia_nemotron_3_5_lightning_70b_custom' }}
        />
      );
      expect(screen.queryByText(/known tool-calling failures/)).not.toBeInTheDocument();
      rerender(
        <InsightsModelPairFields
          {...props}
          {...{ [`${role}Model`]: 'default/nvidia-nemotron-4-lightning-30b' }}
        />
      );
      expect(screen.queryByText(/known tool-calling failures/)).not.toBeInTheDocument();
      rerender(
        <InsightsModelPairFields
          {...props}
          stored={undefined}
          {...{ [`${role}Model`]: lightning }}
        />
      );
      expect(screen.getByText(/known tool-calling failures/)).toHaveTextContent(
        'Try Nemotron 3.5 Super VL instead.'
      );
      expect(props.onDefaultModelChange).not.toHaveBeenCalled();
      expect(props.onFastModelChange).not.toHaveBeenCalled();
    }
  );
});
