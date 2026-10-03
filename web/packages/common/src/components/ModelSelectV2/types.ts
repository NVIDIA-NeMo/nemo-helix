// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelFilterControls } from '@nemo/common/src/api/models/modelFilters';
import type { ModelWorkspaceGroup } from '@nemo/common/src/api/models/useModels';
import type { InferenceParams, ModelEntity } from '@nemo/sdk/generated/platform/schema';

export interface ModelSelection {
  /** Model URN (e.g. "workspace/model_name") */
  model: string;
  /** Adapter name, if an adapter was selected instead of the base model */
  adapter?: string;
  /**
   * The catalogue entry the user picked, when the selection came from the dropdown. Saves
   * callers a lookup (or a second request) for fields like `model_providers`; absent when the
   * selection was restored from a URL, a form default, or any other source outside the list.
   */
  entity?: ModelEntity;
}

export interface ModelSelectV2Props {
  /** Currently selected model (and optional adapter) */
  value: ModelSelection | null;
  /** Called when user selects a model or adapter */
  onValueChange: (selection: ModelSelection) => void;
  /**
   * The page of models to show, grouped by workspace. The dropdown renders exactly what it is
   * given — pair with {@link onSearchChange} and {@link onLoadMore} (see `useModelSearch`) so a
   * large catalogue arrives one page at a time instead of all at once.
   */
  groups?: ModelWorkspaceGroup[];
  /**
   * Server-side search. When set, the dropdown stops filtering {@link groups} itself and instead
   * reports the debounced filter text; the caller re-queries and passes back new `groups`.
   * Leave unset to filter the given groups client-side.
   */
  onSearchChange?: (search: string) => void;
  /** How long the filter box settles before {@link onSearchChange} fires. */
  searchDebounceMs?: number;
  /** Called as the list scrolls near its end. Ignored unless {@link hasMore} is true. */
  onLoadMore?: () => void | Promise<void>;
  /** Whether another page is available for {@link onLoadMore} to fetch. */
  hasMore?: boolean;
  /** Whether the next page is in flight. */
  isLoadingMore?: boolean;
  /** Footer copy shown once every page has loaded. Omit to render nothing. */
  doneLoadingMessage?: string;
  /** Copy shown when no models match. */
  emptyMessage?: string;
  /** Whether the first page of models is still loading */
  loading?: boolean;
  /** Whether the component is disabled */
  disabled?: boolean;
  /** Placeholder text for the model trigger button */
  placeholder?: string;
  /** Show the Custom/Base segmented control toggle */
  showModelTypeToggle?: boolean;
  /** Default active segment when toggle is shown */
  defaultModelType?: ModelType;
  /**
   * Called when the model-type segment changes. Same contract as {@link onSearchChange}: when set,
   * the caller owns the filter and the dropdown stops applying it to {@link groups}.
   */
  onModelTypeChange?: (modelType: ModelType) => void;
  /** Provider, family and size filters. Spread from `useModelSearch`; omit to hide the filter bar. */
  modelFilters?: ModelFilterControls;
  /** Show the params button alongside the model button */
  showParams?: boolean;
  /**
   * Hide each model's adapter sub-list. When true, models render as flat,
   * directly-selectable items even if they have adapters.
   *
   * Use this when only base models are valid selections (e.g. fine-tuning
   * source models, where the customizer requires a model URN with a
   * fileset, not an adapter).
   */
  hideAdapters?: boolean;
  /** Make the component fill the width of its container */
  fullWidth?: boolean;
  /** Preferred side for the dropdown content */
  dropdownSide?: 'top' | 'bottom';
  /** Current inference parameter values */
  inferenceParams?: Partial<InferenceParams>;
  /** Called when the user changes any inference parameter */
  onInferenceParamsChange?: (params: Partial<InferenceParams>) => void;
  /** Called when the model dropdown opens or closes */
  onOpenChange?: (open: boolean) => void;
  /** aria-label for the button group */
  'aria-label'?: string;
}

export type ModelType = 'custom' | 'base';
