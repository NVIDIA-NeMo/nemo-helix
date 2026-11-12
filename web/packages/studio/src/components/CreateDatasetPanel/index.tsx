// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Source: Linear ASTD-896 'Create Dataset — Form Validation and Behavior' handoff; section numbers in
// comments refer to it.

import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import {
  Button,
  SidePanel,
  TabsContent,
  TabsList,
  TabsRoot,
  TabsTrigger,
} from '@nvidia/foundations-react-core';
import { buildCreateRequest } from '@studio/components/CreateDatasetPanel/buildCreateRequest';
import { TAB_LABELS, TAB_ORDER, type TabId } from '@studio/components/CreateDatasetPanel/constants';
import { getDatasetNameError } from '@studio/components/CreateDatasetPanel/datasetName';
import {
  createInitialState,
  draftEntries,
  panelReducer,
} from '@studio/components/CreateDatasetPanel/draft';
import { SourceTabForm } from '@studio/components/CreateDatasetPanel/SourceTabForm';
import { UploadTabForm } from '@studio/components/CreateDatasetPanel/UploadTabForm';
import { useCreateDataset } from '@studio/components/CreateDatasetPanel/useCreateDataset';
import { useWorkspaceSecrets } from '@studio/components/CreateDatasetPanel/useWorkspaceSecrets';
import {
  buildRequestInput,
  hasErrors,
  validateDraft,
  visibleErrors,
} from '@studio/components/CreateDatasetPanel/validateDraft';
import { useFilesetNameStatus } from '@studio/components/filesets/hooks/useFilesetNameStatus';
import { useEffect, useReducer } from 'react';

export interface CreateDatasetPanelProps {
  open: boolean;
  workspace: string;
  onClose: () => void;
  onCreated: (fileset: FilesetOutput) => void;
}

const isTabId = (value: string): value is TabId => TAB_ORDER.some((tab) => tab === value);

/** Side panel that creates a `dataset`-purpose fileset from an external source or uploaded files. */
export function CreateDatasetPanel({
  open,
  workspace,
  onClose,
  onCreated,
}: CreateDatasetPanelProps) {
  const [state, dispatch] = useReducer(panelReducer, undefined, createInitialState);
  const { tab, submitErrors, takenNames } = state;
  const draft = state.drafts[tab];

  const secrets = useWorkspaceSecrets(workspace, open);
  const { submit, isPending, createdName, forgetCreated } = useCreateDataset(workspace);

  // Closing by any route (Cancel, the header X, the host) discards every tab's draft (spec 2).
  useEffect(() => {
    if (!open) {
      dispatch({ type: 'reset' });
      forgetCreated();
    }
  }, [open, forgetCreated]);

  // The spec's own rules decide validity; the server is asked only about names that pass them.
  const formatError = getDatasetNameError(draft.name, { taken: false, workspace });
  const nameStatus = useFilesetNameStatus(workspace, formatError ? '' : draft.name, {
    ownName: createdName,
  });
  const nameTaken = nameStatus === 'conflict' || takenNames.includes(draft.name);

  const errors = validateDraft(tab, draft, {
    workspace,
    nameTaken,
    noSecrets: !secrets.isLoading && secrets.names.length === 0,
  });
  const shownErrors = visibleErrors(errors, draft, submitErrors);

  const handleCreate = async () => {
    const input = hasErrors(errors) ? undefined : buildRequestInput(tab, draft);
    if (!input) {
      dispatch({ type: 'submit-rejected' });
      return;
    }

    const outcome = await submit(
      buildCreateRequest(input),
      tab === 'upload' ? draftEntries(draft) : undefined
    );
    if (outcome.status === 'conflict') {
      dispatch({ type: 'name-taken', name: draft.name });
    } else if (outcome.status === 'created') {
      onCreated(outcome.fileset);
      dispatch({ type: 'reset' });
      forgetCreated();
      onClose();
    }
  };

  return (
    <SidePanel
      slotHeading="New Dataset"
      side="right"
      open={open}
      onOpenChange={(next) => {
        if (!next) onClose();
      }}
      className="max-w-[600px] w-full"
      bordered
      modal
      // The draft is unsaved work, so a stray click outside must not discard it (spec 2).
      closeOnClickOutside={false}
      slotFooter={
        <>
          <Button type="button" kind="tertiary" onClick={onClose}>
            Cancel
          </Button>
          <Button
            type="button"
            color="brand"
            // A manually edited, invalid name disables Create; every other problem is reported on submit (spec 11.2).
            disabled={isPending || (draft.nameTouched && Boolean(errors.name))}
            onClick={handleCreate}
          >
            {isPending ? 'Creating…' : 'Create Dataset'}
          </Button>
        </>
      }
    >
      <TabsRoot
        className="w-full min-w-0"
        value={tab}
        onValueChange={(next) => {
          if (isTabId(next)) dispatch({ type: 'tab-selected', tab: next });
        }}
      >
        <TabsList>
          {TAB_ORDER.map((id) => (
            // The hidden spacing copy of the label (reserves bold width) would double its text.
            <TabsTrigger key={id} value={id} renderSpacingElement={false}>
              {TAB_LABELS[id]}
            </TabsTrigger>
          ))}
        </TabsList>
        {/* Only the active tab renders: the other drafts live in the reducer. */}
        <TabsContent className="w-full min-w-0 p-0 items-stretch" value={tab}>
          {tab === 'upload' ? (
            <UploadTabForm draft={draft} errors={shownErrors} dispatch={dispatch} />
          ) : (
            <SourceTabForm
              key={tab}
              tab={tab}
              draft={draft}
              errors={shownErrors}
              secrets={secrets}
              dispatch={dispatch}
            />
          )}
        </TabsContent>
      </TabsRoot>
    </SidePanel>
  );
}
