// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { InsightsModelPairFields } from '@studio/components/ImportTracesModal/InsightsModelPairFields';
import { render, screen } from '@testing-library/react';

vi.mock('@nemo/common/src/components/ModelSelectV2', () => ({
  WorkspaceModelSelect: ({ value }: { value: { model: string } | null }) => (
    <span>{value?.model}</span>
  ),
}));

describe('Compass model guidance', () => {
  it.each(['default', 'fast'] as const)(
    'warns for the effective %s model and clears on override',
    (role) => {
      const lightning = 'default/Nemotron-3.5-Lightning-30B';
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
          stored={undefined}
          {...{ [`${role}Model`]: lightning }}
        />
      );
      expect(screen.getByText(/known tool-calling failures/)).toHaveTextContent('You can continue');
      expect(props.onDefaultModelChange).not.toHaveBeenCalled();
      expect(props.onFastModelChange).not.toHaveBeenCalled();
    }
  );
});
