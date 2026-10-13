// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { strategyOptions } from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizationStrategySelect/constants';

const routing = (installed: readonly string[] | undefined) =>
  strategyOptions(installed).find((option) => option.id === 'routing');

describe('strategyOptions', () => {
  it('enables routing when the platform has the switchyard strategy', () => {
    expect(routing(['legacy', 'switchyard'])).toMatchObject({ enabled: true });
    expect(routing(['legacy', 'switchyard'])?.tag).toBeUndefined();
  });

  it('disables routing and says why when switchyard is not installed', () => {
    expect(routing(['legacy'])).toMatchObject({
      enabled: false,
      tag: { label: 'Not installed' },
    });
  });

  it('disables routing without a tag while the listing is unknown', () => {
    expect(routing(undefined)).toMatchObject({ enabled: false });
    expect(routing(undefined)?.tag).toBeUndefined();
  });

  it('leaves strategies that need no plugin alone', () => {
    const byId = Object.fromEntries(strategyOptions([]).map((option) => [option.id, option]));
    expect(byId.hyperparameter.enabled).toBe(true);
    expect(byId.upload.enabled).toBe(true);
    expect(byId.skill.enabled).toBe(false);
  });
});
