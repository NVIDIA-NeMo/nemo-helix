// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getGuardrailsGetGuardrailConfigQueryKey,
  useGuardrailsUpdateConfig,
} from '@nemo/sdk/generated/platform/guardrails';
import type { GuardrailConfig } from '@nemo/sdk/generated/platform/schema';
import {
  GuardrailFormContext,
  type GuardrailFormContextValue,
} from '@studio/routes/guardrails/GuardrailForm/context';
import {
  applyFormToConfig,
  type GuardrailFormValues,
  guardrailDraftKey,
  guardrailFormSchema,
  mapConfigToForm,
  type StoredDraft,
} from '@studio/routes/guardrails/GuardrailForm/formModel';
import { useLocalStorage } from '@studio/util/hooks/useLocalStorage';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, type ReactNode, useCallback, useEffect, useMemo, useRef } from 'react';
import { FormProvider, useForm, useWatch } from 'react-hook-form';

export const GuardrailFormProvider: FC<{ config: GuardrailConfig; children: ReactNode }> = ({
  config,
  children,
}) => {
  const workspace = config.workspace;
  const name = config.name ?? '';
  const baseVersion = config.updated_at ?? '';

  const serverValues = useMemo(() => mapConfigToForm(config.data), [config.data]);

  const form = useForm<GuardrailFormValues>({
    resolver: zodResolver(guardrailFormSchema),
    defaultValues: serverValues,
  });

  const [stored, setStored, clearStored] = useLocalStorage<StoredDraft>(
    guardrailDraftKey(workspace, name)
  );

  // Hydrate a persisted, non-stale draft once. defaultValues stay the server
  // values, so `shouldDirty` makes isDirty reflect draft-vs-server correctly.
  const hydratedRef = useRef(false);
  useEffect(() => {
    if (hydratedRef.current) {
      return;
    }
    hydratedRef.current = true;
    if (!stored) {
      return;
    }
    if (stored.baseVersion !== baseVersion) {
      clearStored(); // branched from an older server version — discard.
      return;
    }
    for (const key of Object.keys(stored.values) as (keyof GuardrailFormValues)[]) {
      form.setValue(key, stored.values[key], { shouldDirty: true });
    }
  }, [stored, baseVersion, clearStored, form]);

  // Persist edits (keyed off RHF's dirty state so a saved form never re-persists).
  const values = useWatch({ control: form.control });
  const { isDirty } = form.formState;
  useEffect(() => {
    if (isDirty) {
      setStored({ baseVersion, values: form.getValues() });
    } else {
      clearStored();
    }
  }, [isDirty, values, baseVersion, form, setStored, clearStored]);

  const queryClient = useQueryClient();
  const toast = useToast();
  const { mutateAsync: updateConfig, isPending: isSaving } = useGuardrailsUpdateConfig();

  const save = useMemo(
    () =>
      form.handleSubmit(async (submitted) => {
        const data = applyFormToConfig(submitted);
        try {
          await updateConfig({ workspace, name, data: { data: { ...data } } });
        } catch {
          toast.error('Failed to save the guardrail. Please try again.');
          return;
        }
        // The PATCH landed — the save is complete. Commit the local state and
        // report success regardless of what the cache refresh below does.
        clearStored();
        form.reset(submitted); // new baseline = saved values → isDirty false.
        toast.success('Guardrail saved.');
        // Refresh the cached config; a refetch failure doesn't undo the save,
        // and the cache resyncs on the next fetch, so it stays silent.
        try {
          await queryClient.invalidateQueries({
            queryKey: getGuardrailsGetGuardrailConfigQueryKey(workspace, name),
          });
        } catch {
          // ignore — save already succeeded
        }
      }),
    [form, updateConfig, workspace, name, queryClient, clearStored, toast]
  );

  const resetToServer = useCallback(() => {
    clearStored();
    form.reset(serverValues);
  }, [clearStored, form, serverValues]);

  const value = useMemo<GuardrailFormContextValue>(
    () => ({ config, save, isSaving, resetToServer }),
    [config, save, isSaving, resetToServer]
  );

  return (
    <GuardrailFormContext.Provider value={value}>
      <FormProvider {...form}>{children}</FormProvider>
    </GuardrailFormContext.Provider>
  );
};
