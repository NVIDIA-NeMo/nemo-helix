// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NvidiaGuiIconProps } from '@nemo/icons';
import { Upload, Puzzle, Route, Sliders } from '@nemo/icons/icons';
import type { OptimizationStrategy } from '@nemo/sdk/generated/agent-optimization/schema/OptimizationStrategy';
import { STUDIO_OPTIMIZE_STRATEGY } from '@studio/api/agents/useLaunchOptimizeStudy';
import type { StartOption } from '@studio/components/StartOptions/types';
import type { FC } from 'react';

/** The `run-strategy` name the Switchyard plugin registers its strategy under. */
export const SWITCHYARD_STRATEGY = 'switchyard';

/** Icons for the strategies Studio has a form for; every other strategy gets {@link Puzzle}. */
const STRATEGY_ICONS: Record<string, FC<Omit<NvidiaGuiIconProps, 'iconName'>>> = {
  [STUDIO_OPTIMIZE_STRATEGY]: Sliders,
  [SWITCHYARD_STRATEGY]: Route,
};

/** Tile id for uploading a `config-tuning` bundle; not a valid strategy name, so it cannot collide. */
export const UPLOAD_CONFIG_OPTION_ID = 'upload a config';

/**
 * The one tile that is not a strategy of its own: a bring-your-own bundle for `config-tuning`,
 * beside that strategy's guided form.
 */
const UPLOAD_CONFIG_OPTION: StartOption = {
  id: UPLOAD_CONFIG_OPTION_ID,
  title: 'Upload a config',
  description:
    'Bring your own optimize bundle, from your computer or a fileset: the config plus the dataset and any other files it references.',
  icon: Upload,
  enabled: true,
};

/**
 * One tile per strategy the platform has installed, titled and described by the plugin that ships
 * it, so installing a strategy is all it takes to offer it. Tile ids are the strategy names. The
 * upload tile leads when `config-tuning` is installed, since that is the strategy it submits to.
 */
export const strategyOptions = (strategies: readonly OptimizationStrategy[]): StartOption[] => {
  const options = strategies.map((strategy) => ({
    id: strategy.name,
    title: strategy.title || strategy.name,
    description: strategy.description ?? '',
    icon: STRATEGY_ICONS[strategy.name] ?? Puzzle,
    enabled: true,
  }));
  const canUpload = strategies.some((strategy) => strategy.name === STUDIO_OPTIMIZE_STRATEGY);
  return canUpload ? [UPLOAD_CONFIG_OPTION, ...options] : options;
};
