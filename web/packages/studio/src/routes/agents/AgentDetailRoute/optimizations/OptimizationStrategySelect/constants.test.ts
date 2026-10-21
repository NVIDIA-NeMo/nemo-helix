// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  strategyOptions,
  UPLOAD_CONFIG_OPTION_ID,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizationStrategySelect/constants';
import { Puzzle, Route, SlidersHorizontal } from 'lucide-react';

describe('strategyOptions', () => {
  it('offers each installed strategy in its own words, keyed by name', () => {
    expect(
      strategyOptions([
        {
          name: 'prompt-master',
          title: 'Prompt optimization using Prompt Master',
          description: 'Rewrite the system prompt.',
          requires_config: false,
        },
      ])
    ).toEqual([
      {
        id: 'prompt-master',
        title: 'Prompt optimization using Prompt Master',
        description: 'Rewrite the system prompt.',
        icon: Puzzle,
        enabled: true,
      },
    ]);
  });

  it('gives the strategies Studio has forms for their own icons', () => {
    const icons = strategyOptions([
      { name: 'config-tuning', title: 'Hyper-parameter and prompt optimization' },
      { name: 'switchyard', title: 'Model routing using Switchyard' },
    ])
      .filter((option) => option.id !== UPLOAD_CONFIG_OPTION_ID)
      .map((option) => option.icon);
    expect(icons).toEqual([SlidersHorizontal, Route]);
  });

  it('leads with the upload tile only when config-tuning is installed', () => {
    expect(
      strategyOptions([{ name: 'config-tuning', title: 'Tune' }]).map((option) => option.id)
    ).toEqual([UPLOAD_CONFIG_OPTION_ID, 'config-tuning']);
    expect(
      strategyOptions([{ name: 'switchyard', title: 'Route' }]).map((option) => option.id)
    ).toEqual(['switchyard']);
  });

  it('falls back to the name when a strategy has no title', () => {
    expect(strategyOptions([{ name: 'bare', title: '' }])[0]?.title).toBe('bare');
  });
});
