// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { IntakeTelemetryStatusBadge } from '@studio/components/IntakeDetail/IntakeComponents/IntakeTelemetryStatusBadge';
import { render, screen } from '@studio/tests/util/render';

describe('IntakeTelemetryStatusBadge', () => {
  it.each([
    ['success', 'Completed'],
    ['error', 'Error'],
    ['cancelled', 'Cancelled'],
    ['unknown', 'Unknown'],
    [undefined, 'Unknown'],
  ] as const)('renders %s as %s', (status, label) => {
    render(<IntakeTelemetryStatusBadge status={status} />);

    expect(screen.getByText(label)).toBeInTheDocument();
  });
});
