// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelSelection } from '@nemo/common/src/components/ModelSelectV2/types';
import type { AddColumnSelection } from '@studio/components/AddColumnPalette/types';
import type { FilesetTemplate } from '@studio/components/CreateFilesetStart/types';
import {
  type BuilderColumn,
  buildColumnsFromTemplate,
  defaultColumnName,
  defaultColumnValues,
  findColumnOption,
} from '@studio/routes/DataDesignerJobBuildRoute/columns';
import {
  type BuilderModel,
  buildModelsFromTemplate,
  builderModelFromSelection,
  fetchAutoFillCandidates,
  findWorkspaceModel,
  firstAvailableModel,
} from '@studio/routes/DataDesignerJobBuildRoute/models';
import { useCallback, useEffect, useRef, useState } from 'react';
import { type UseFormReturn, useFieldArray, useForm } from 'react-hook-form';

/** Which palette the left aside shows. */
export type PaletteTab = 'columns' | 'models';

/** All mutable values in the build route, owned by React Hook Form. */
export interface JobBuilderFormValues {
  name: string;
  rows: string;
  columns: BuilderColumn[];
  models: BuilderModel[];
}

export interface JobBuilderValues {
  columns: BuilderColumn[];
  models: BuilderModel[];
  name: string;
  rows: string;
}

/**
 * Fully-formed initial builder state used to clone an existing job. When present it seeds the
 * form instead of the template, so the canvas opens pre-filled with the source job's schema.
 */
export interface JobBuilderSeed {
  name: string;
  rows: string;
  columns: BuilderColumn[];
  models: BuilderModel[];
}

/**
 * A template model auto-fill could not honour: the workspace does not serve `requested`. The
 * requested name stays in the form so the recipe still reads as the template wrote it; the build
 * route surfaces the issue and blocks the run until the user picks a model that exists.
 */
export interface TemplateModelIssue {
  /** Builder model id, stable across alias edits — how the issue is matched to live form state. */
  id: string;
  alias: string;
  requested: string;
}

/** Outcome of a single model's auto-fill lookup: either it ran (with or without a match), or the
 *  request itself rejected. */
interface ModelLookupResult {
  rejected: boolean;
  resolved: { model: string; provider: string } | null;
}

/**
 * The issues that still apply: auto-fill records them once, but the user can resolve one by
 * picking an available model, which fills in the provider auto-fill could not.
 */
export const unresolvedTemplateModelIssues = (
  issues: TemplateModelIssue[],
  models: BuilderModel[]
): TemplateModelIssue[] =>
  issues.filter((issue) => models.some((model) => model.id === issue.id && !model.provider));

/**
 * Column/model state for the recipe builder. Selecting a column and selecting a model are
 * mutually exclusive — only one config panel shows at a time.
 *
 * Job-level concerns (name, row count, validation, preview, submit) live in the route so
 * this hook stays a pure graph-editing store.
 *
 * A template's seeded models are auto-filled once from `workspace`, resolving each spec with a
 * targeted lookup rather than the whole model catalogue.
 *
 * `seed`, when provided (cloning a job), takes precedence over the template and pre-fills the
 * form with the source job's columns, models, name, and row count.
 */
export const useJobBuilder = (
  template: FilesetTemplate | null,
  workspace: string,
  seed: JobBuilderSeed | null = null
) => {
  // Seed once from the clone source, else the template (if any). `useForm` keeps these values
  // outside the route's render cycle, so a field edit notifies only subscribing components.
  const initialColumns = useRef(
    seed ? seed.columns : template ? buildColumnsFromTemplate(template.columns) : []
  );
  const initialModels = useRef(seed ? seed.models : buildModelsFromTemplate(template?.models));
  const form = useForm<JobBuilderFormValues>({
    defaultValues: {
      name: seed?.name ?? template?.id ?? 'untitled-dataset',
      rows: seed?.rows ?? '100',
      columns: initialColumns.current,
      models: initialModels.current,
    },
  });
  const { getValues, setValue } = form;
  const {
    fields: columnFields,
    append: appendColumn,
    remove: removeColumnField,
  } = useFieldArray({
    control: form.control,
    name: 'columns',
    keyName: 'fieldId',
  });

  const [selectedId, setSelectedId] = useState<string | null>(null);
  // Set only when a column is added, so the canvas centers new nodes but not clicked ones.
  const [focusId, setFocusId] = useState<string | null>(null);
  // Continue numbering after any preloaded template columns so ids stay unique.
  const nextId = useRef(initialColumns.current.length);

  // The models referenced by LLM columns via `model_alias`; part of the same job config.
  // Seeded once from the template (if any); providers/models are auto-filled below.
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const nextModelId = useRef(initialModels.current.length);
  const [paletteTab, setPaletteTab] = useState<PaletteTab>('columns');

  const [templateModelIssues, setTemplateModelIssues] = useState<TemplateModelIssue[]>([]);
  const [autoFillError, setAutoFillError] = useState(false);
  const autoFilled = useRef(false);

  const runAutoFill = useCallback(() => {
    if (!workspace) return;
    const pending = getValues('models').filter((model) => !model.provider);
    if (pending.length === 0) return;
    autoFilled.current = true;
    setAutoFillError(false);

    void (async () => {
      const resolutions: (readonly [string, ModelLookupResult])[] = await Promise.all(
        pending.map(async (model) => {
          const preferred = model.model || undefined;
          try {
            const candidates = await fetchAutoFillCandidates(workspace, preferred);
            const resolved = preferred
              ? findWorkspaceModel(candidates, preferred)
              : firstAvailableModel(candidates);
            return [model.id, { rejected: false, resolved }] as const;
          } catch {
            return [model.id, { rejected: true, resolved: null }] as const;
          }
        })
      );
      const byId = new Map(resolutions);
      const anyRejected = resolutions.some(([, result]) => result.rejected);
      if (anyRejected) autoFilled.current = false;
      setAutoFillError(anyRejected);
      setTemplateModelIssues(
        pending.flatMap((model) => {
          const result = byId.get(model.id);
          if (!result || result.rejected) return [];
          return model.model && !result.resolved
            ? [{ id: model.id, alias: model.alias, requested: model.model }]
            : [];
        })
      );
      setValue(
        'models',
        getValues('models').map((model) => {
          const resolved = byId.get(model.id)?.resolved;
          return resolved
            ? { ...model, model: resolved.model, provider: resolved.provider }
            : model;
        })
      );
    })();
  }, [getValues, setValue, workspace]);

  useEffect(() => {
    if (autoFilled.current) return;
    runAutoFill();
  }, [runAutoFill]);

  const selectColumn = useCallback((id: string | null) => {
    setSelectedId(id);
    if (id !== null) setSelectedModelId(null);
  }, []);
  const selectModel = useCallback((id: string | null) => {
    setSelectedModelId(id);
    if (id !== null) setSelectedId(null);
  }, []);

  const handleAddColumn = useCallback(
    (selection: AddColumnSelection) => {
      const columns = getValues('columns');
      if (
        selection.columnType === 'seed-dataset' &&
        columns.some((column) => column.option.columnType === 'seed-dataset')
      ) {
        return;
      }
      const option = findColumnOption(selection);
      if (!option) return;
      const id = `col-${nextId.current++}`;
      const name = defaultColumnName(option, new Set(columns.map((column) => column.name)));
      appendColumn({ id, option, name, values: defaultColumnValues(option) });
      selectColumn(id);
      setFocusId(id);
    },
    [appendColumn, getValues, selectColumn]
  );

  const removeColumn = useCallback(
    (id: string) => {
      const index = getValues('columns').findIndex((column) => column.id === id);
      if (index >= 0) removeColumnField(index);
      setSelectedId((current) => (current === id ? null : current));
    },
    [getValues, removeColumnField]
  );

  const handleAddModel = useCallback(
    (selection: ModelSelection, provider: string) => {
      const id = `model-${nextModelId.current++}`;
      const models = getValues('models');
      setValue('models', [
        ...models,
        builderModelFromSelection(
          id,
          selection,
          provider,
          new Set(models.map((model) => model.alias.trim()))
        ),
      ]);
      selectModel(id);
    },
    [getValues, selectModel, setValue]
  );

  const removeModel = useCallback(
    (id: string) => {
      setValue(
        'models',
        getValues('models').filter((model) => model.id !== id)
      );
      setSelectedModelId((current) => (current === id ? null : current));
    },
    [getValues, setValue]
  );

  const getBuilderValues = useCallback((): JobBuilderValues => {
    const values = getValues();
    return {
      name: values.name,
      rows: values.rows,
      columns: values.columns,
      models: values.models,
    };
  }, [getValues]);

  return {
    form: form as UseFormReturn<JobBuilderFormValues>,
    getBuilderValues,
    columnCount: columnFields.length,
    selectedColumnId: selectedId,
    selectedModelId,
    focusId,
    templateModelIssues,
    autoFillError,
    retryAutoFill: runAutoFill,
    paletteTab,
    setPaletteTab,
    selectColumn,
    selectModel,
    handleAddColumn,
    removeColumn,
    handleAddModel,
    removeModel,
  };
};
