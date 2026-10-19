// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { BASIC_ALL_MODELS_DROPDOWN_FILTER } from '@nemo/common/src/api/models/useModels';
import { useModelsFromWorkspace } from '@nemo/common/src/api/models/useModelsFromWorkspace';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { ENTITY_NAME_HELP, entityNameSchema, toCopyName } from '@nemo/common/src/utils/entityName';
import { useGuardrailsCreateConfig } from '@nemo/sdk/generated/platform/guardrails';
import type {
  GuardrailConfig,
  GuardrailConfigInput,
  GuardrailConfigInputData,
  RailsConfig,
} from '@nemo/sdk/generated/platform/schema';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { resolveDefaultGuardrailModel } from '@studio/routes/guardrails/defaultModel';
import { setMainModelName } from '@studio/routes/guardrails/GuardrailConfigTab/mainModel';
import { getGuardrailDetailRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useEffect, useRef } from 'react';
import { useForm } from 'react-hook-form';
import { useNavigate } from 'react-router';
import { z } from 'zod';

const createGuardrailFormSchema = z.object({
  name: entityNameSchema(),
});

type FormData = z.infer<typeof createGuardrailFormSchema>;

interface Props {
  open: boolean;
  onClose: () => void;
  /** When set, the modal duplicates this config instead of creating an empty one. */
  sourceConfig?: GuardrailConfig;
}

export const CreateGuardrailModal: FC<Props> = ({ open, onClose, sourceConfig }) => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const isDuplicate = Boolean(sourceConfig);
  const defaultName = sourceConfig?.name ? toCopyName(sourceConfig.name) : '';

  // Only for a fresh config — a duplicate inherits the source's models below. Held to the
  // open modal so closing it doesn't leave a query running.
  const { groups } = useModelsFromWorkspace({
    workspace: workspace ?? null,
    query: BASIC_ALL_MODELS_DROPDOWN_FILTER,
    queryOptions: { enabled: open && !isDuplicate },
  });

  const {
    control,
    handleSubmit,
    reset,
    formState: { isValid },
  } = useForm<FormData>({
    resolver: zodResolver(createGuardrailFormSchema),
    defaultValues: { name: defaultName },
    mode: 'onChange',
  });

  const {
    mutateAsync: createConfig,
    isPending,
    error,
    reset: resetMutation,
  } = useGuardrailsCreateConfig();

  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    // The modal can stay mounted across opens, so seed the name for this open.
    reset({ name: defaultName });
    // setTimeout 0 lets the dialog's built-in focus management run first (it would otherwise
    // focus the slotInfo icon button, which appears before the input in DOM order).
    const id = setTimeout(() => {
      const input = containerRef.current?.querySelector<HTMLInputElement>('input');
      input?.focus();
      input?.select();
    }, 0);
    return () => clearTimeout(id);
  }, [open, defaultName, reset]);

  const handleClose = () => {
    reset({ name: defaultName });
    resetMutation();
    onClose();
  };

  const onSubmit = async (data: FormData) => {
    const payload: GuardrailConfigInput = { name: data.name };
    if (sourceConfig) {
      if (sourceConfig.description) payload.description = sourceConfig.description;
      if (sourceConfig.data) payload.data = { ...sourceConfig.data } as GuardrailConfigInputData;
    } else {
      // Seed the main model so the new config can run its tests without a detour through
      // the Configuration tab. Resolution returning null (no models served, or none with a
      // provider) creates the config without `models` — the tab's field then shows empty,
      // which beats writing a name that fails at run time.
      const model = resolveDefaultGuardrailModel(groups);
      if (model) {
        const railsConfig: RailsConfig = { models: setMainModelName(undefined, model) };
        payload.data = railsConfig as GuardrailConfigInputData;
      }
    }

    let config: GuardrailConfig;
    try {
      config = await createConfig({ workspace, data: payload });
    } catch {
      // The modal stays open and surfaces the failure via `errorText` below.
      return;
    }
    await queryClient.invalidateQueries({
      queryKey: [`/apis/guardrails/v2/workspaces/${workspace}/configs`],
    });
    handleClose();
    navigate(getGuardrailDetailRoute(workspace, config.name ?? data.name));
  };

  return (
    <FormModal
      open={open}
      title={isDuplicate ? 'Duplicate Guardrail' : 'Create Guardrail'}
      submitButtonText="Create"
      disabled={isPending}
      loading={isPending}
      submitDisabled={!isValid}
      errorText={error ? getErrorMessage(error) : null}
      onSubmit={handleSubmit(onSubmit)}
      onClose={handleClose}
    >
      <div ref={containerRef}>
        <ControlledTextInput
          label="Name"
          disabled={isPending}
          useControllerProps={{ name: 'name', control }}
          formFieldProps={{ slotInfo: ENTITY_NAME_HELP }}
          attributes={{
            Input: {
              autoComplete: 'off',
              autoCapitalize: 'none',
              autoCorrect: 'off',
              spellCheck: false,
            },
          }}
        />
      </div>
    </FormModal>
  );
};
