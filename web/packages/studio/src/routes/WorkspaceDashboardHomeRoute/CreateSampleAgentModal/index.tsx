// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { WorkspaceModelSelect } from '@nemo/common/src/components/ModelSelectV2';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { hasModelProvider } from '@nemo/common/src/utils/models';
import { getEntitiesListWorkspacesQueryKey } from '@nemo/sdk/generated/platform/entity-store';
import { FormField } from '@nvidia/foundations-react-core';
import { useRecentWorkspaces } from '@studio/components/WorkspaceDropdown/useRecentWorkspaces';
import { useOidcBearerToken } from '@studio/providers/auth/useOidcBearerToken';
import { getWorkspaceDashboardRoute } from '@studio/routes/utils';
import { streamCreateSampleAgent } from '@studio/routes/WorkspaceDashboardHomeRoute/CreateSampleAgentModal/streamCreateSampleAgent';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { type FC, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router';

const MODEL_HELP =
  "The sample agent's results depend on this model. Pick a strong, tool-calling chat model for the best results.";

export interface CreateSampleAgentModalProps {
  readonly open: boolean;
  readonly onClose: () => void;
  /** Workspace whose model catalogue the picker searches. */
  readonly workspace: string;
}

/** Mount only while open: state (model, error, in-flight stream) lives for one open. */
export const CreateSampleAgentModal: FC<CreateSampleAgentModalProps> = ({
  open,
  onClose,
  workspace,
}) => {
  const toast = useToast();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const accessToken = useOidcBearerToken();
  const { addRecentWorkspace } = useRecentWorkspaces();

  const [model, setModel] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const {
    mutate: createSample,
    error,
    isPending,
  } = useMutation({
    mutationFn: (selectedModel: string) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      return streamCreateSampleAgent({ model: selectedModel }, accessToken, controller.signal);
    },
    onSuccess: (result) => {
      toast.success(
        result.status === 'created'
          ? `Sample workspace "${result.workspace}" created`
          : `Opened your existing sample workspace "${result.workspace}"`
      );
      void queryClient.invalidateQueries({ queryKey: getEntitiesListWorkspacesQueryKey() });
      onClose();
      // The workspace dropdown labels from recents when the new workspace isn't in its list yet.
      addRecentWorkspace(result.workspace);
      navigate(getWorkspaceDashboardRoute(result.workspace));
    },
  });

  return (
    <FormModal
      open={open}
      onClose={onClose}
      title="Create Sample Workspace"
      instruction="Create a sample workspace that includes the Email Security Triage agent, deployed with the model you choose."
      submitButtonText="Create"
      onSubmit={(e) => {
        e.preventDefault();
        if (model) createSample(model);
      }}
      disabled={isPending}
      loading={isPending}
      submitDisabled={!model}
      errorText={
        error
          ? `Couldn't create the sample workspace: ${getErrorMessage(error, 'Something went wrong.')}`
          : null
      }
    >
      <FormField slotLabel="Model" slotHelp={MODEL_HELP}>
        <WorkspaceModelSelect
          workspace={workspace}
          include={hasModelProvider}
          value={model ? { model } : null}
          onValueChange={({ model: next }) => setModel(next)}
          placeholder="Select a model"
          disabled={isPending}
          hideAdapters
          fullWidth
          aria-label="Model"
        />
      </FormField>
    </FormModal>
  );
};
