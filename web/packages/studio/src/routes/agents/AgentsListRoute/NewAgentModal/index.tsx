// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getAgentsListAgentsQueryKey,
  useAgentsCreateAgent,
} from '@nemo/sdk/generated/agents/agents';
import {
  Button,
  Flex,
  Label,
  Select,
  Stack,
  TabsContent,
  TabsList,
  TabsRoot,
  TabsTrigger,
  Text,
  UploadInputElement,
  UploadRoot,
  UploadTrigger,
} from '@nvidia/foundations-react-core';
import {
  AgentSpecFilesetOrphanError,
  useCreateAgentFromUpload,
} from '@studio/api/agents/useCreateAgentFromUpload';
import { CodingAgentPromptEditor } from '@studio/components/CodingAgentPromptEditor';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { agentIntegrationPrompt } from '@studio/routes/agents/AgentDetailRoute/overview/codingAgentPrompts';
import {
  AGENT_CONFIG_FILENAME,
  uploadAgentFormSchema,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import type {
  NewAgentModalProps,
  NewAgentTab,
  PickedFile,
  UploadAgentEntry,
  UploadAgentFormData,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/type';
import { useTraceAgentNames } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/useTraceAgentNames';
import {
  agentNameFromConfig,
  collectAgentEntries,
  findNonUtf8Path,
  parseAgentConfig,
  pickedFromDataTransfer,
  pickedFromFileList,
  tooManyPickedFiles,
  totalEntryBytes,
  validateAgentEntries,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import {
  type ChangeEventHandler,
  type DragEventHandler,
  type FC,
  useCallback,
  useMemo,
  useRef,
  useState,
} from 'react';
import { type SubmitHandler, useForm, useWatch } from 'react-hook-form';
import { useNavigate } from 'react-router';

export const NewAgentModal: FC<NewAgentModalProps> = ({ open, onClose, workspace }) => {
  const toast = useToast();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const inputRef = useRef<HTMLInputElement>(null);
  const setDirectoryInput = useCallback((node: HTMLInputElement | null) => {
    inputRef.current = node;
    // webkitdirectory is absent from React's input attribute types.
    node?.setAttribute('webkitdirectory', '');
  }, []);
  const [entries, setEntries] = useState<UploadAgentEntry[]>([]);
  const [directoryName, setDirectoryName] = useState('');
  const [selectionError, setSelectionError] = useState<string | undefined>(undefined);
  const [replaceArmedFor, setReplaceArmedFor] = useState<string | null>(null);
  const [tab, setTab] = useState<NewAgentTab>('coding-agent-prompt');
  const [tracedAgent, setTracedAgent] = useState('');

  const {
    mutateAsync: createAgent,
    error: createError,
    isPending,
    reset: resetMutation,
  } = useCreateAgentFromUpload({
    onSuccess: (agent) => {
      toast.success(`Agent "${agent.name}" created`);
      void queryClient.invalidateQueries({ queryKey: getAgentsListAgentsQueryKey(workspace) });
      resetAndClose();
      if (agent.name) navigate(getAgentDetailRoute(workspace, agent.name));
    },
  });

  const {
    control,
    setValue,
    handleSubmit,
    reset: resetForm,
    formState: { errors },
  } = useForm({
    resolver: zodResolver(uploadAgentFormSchema),
    defaultValues: { name: '' },
    disabled: isPending,
    mode: 'onChange',
  });

  const tracedAgents = useTraceAgentNames(workspace, open && tab === 'imported-traces');

  const {
    mutateAsync: createTracedAgent,
    error: tracedCreateError,
    isPending: isCreatingTraced,
    reset: resetTracedMutation,
  } = useAgentsCreateAgent({
    mutation: {
      onSuccess: (agent) => {
        toast.success(`Agent "${agent.name}" created`);
        void queryClient.invalidateQueries({ queryKey: getAgentsListAgentsQueryKey(workspace) });
        resetAndClose();
        if (agent.name) navigate(getAgentDetailRoute(workspace, agent.name));
      },
    },
  });

  // useWatch re-renders this modal on every keystroke; the summary depends only on entries.
  const entriesSummary = useMemo(
    () =>
      entries.length === 0
        ? undefined
        : `${directoryName} — ${entries.length} files, ${Math.max(1, Math.round(totalEntryBytes(entries) / 1000))} KB`,
    [directoryName, entries]
  );

  const watchedName = useWatch({ control, name: 'name' });
  // Derived, not stored: an armed replace targets one fileset, so editing the name
  // disarms it in the same render rather than one render later.
  const replaceOrphan = replaceArmedFor !== null && replaceArmedFor === watchedName?.trim();

  const resetAndClose = () => {
    resetMutation();
    resetTracedMutation();
    resetForm({ name: '' });
    setEntries([]);
    setDirectoryName('');
    setSelectionError(undefined);
    setReplaceArmedFor(null);
    setTab('coding-agent-prompt');
    setTracedAgent('');
    onClose();
  };

  // Directory reads finish out of order, so the newest selection has to win.
  const selectionSeq = useRef(0);
  const beginSelection = (): (() => boolean) => {
    const selection = ++selectionSeq.current;
    // Dropping the entries disables submit until this selection validates.
    resetMutation();
    setEntries([]);
    setSelectionError(undefined);
    setReplaceArmedFor(null);
    return () => selection !== selectionSeq.current;
  };

  // Both entry points land here so a drop is validated exactly like a pick.
  const acceptPicked = async (picked: PickedFile[], superseded: () => boolean) => {
    setDirectoryName(picked[0]?.relativePath.split('/')[0] ?? '');
    const collected = collectAgentEntries(picked);

    const problem = validateAgentEntries(collected);
    if (problem) {
      setEntries([]);
      setSelectionError(problem);
      return;
    }

    const binaryPath = await findNonUtf8Path(collected);
    if (superseded()) return;
    if (binaryPath) {
      setEntries([]);
      setSelectionError(
        `${binaryPath} is not a text file. Agent files are delivered to container deployments as text, so the agent would fail to deploy. Remove it and try again.`
      );
      return;
    }

    const configEntry = collected.find((item) => item.path === AGENT_CONFIG_FILENAME);
    try {
      const config = parseAgentConfig((await configEntry?.file.text()) ?? '');
      if (superseded()) return;
      setValue('name', agentNameFromConfig(config) ?? '', { shouldValidate: true });
    } catch (error) {
      if (superseded()) return;
      setEntries([]);
      setSelectionError(
        getErrorMessage(error as Error) || `Could not read ${AGENT_CONFIG_FILENAME}`
      );
      return;
    }

    setEntries(collected);
  };

  const rejectOversized = (count: number): boolean => {
    const oversized = tooManyPickedFiles(count);
    if (!oversized) return false;
    setEntries([]);
    setDirectoryName('');
    setSelectionError(oversized);
    return true;
  };

  const onDirectoryPicked: ChangeEventHandler<HTMLInputElement> = async (event) => {
    const fileList = event.target.files;
    const pickedCount = fileList?.length ?? 0;
    if (pickedCount === 0) return;

    const superseded = beginSelection();
    if (rejectOversized(pickedCount)) {
      event.target.value = '';
      return;
    }

    const picked = pickedFromFileList(Array.from(fileList ?? []));
    event.target.value = '';
    await acceptPicked(picked, superseded);
  };

  const onDirectoryDropped: DragEventHandler<HTMLLabelElement> = async (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (isPending) return;

    const items = Array.from(event.dataTransfer.items);
    if (items.length === 0) return;

    const superseded = beginSelection();
    const picked = await pickedFromDataTransfer(items);
    if (superseded()) return;

    if (picked.length === 0) {
      setSelectionError('That drop contained no readable files.');
      return;
    }
    if (rejectOversized(picked.length)) return;

    await acceptPicked(picked, superseded);
  };

  const onSubmit: SubmitHandler<UploadAgentFormData> = async (formData) => {
    const name = formData.name.trim();
    try {
      await createAgent({ workspace, name, entries, replaceOrphanedFileset: replaceOrphan });
    } catch (error) {
      // An orphaned fileset is recoverable, so the next submit replaces it.
      setReplaceArmedFor(error instanceof AgentSpecFilesetOrphanError ? name : null);
    }
  };

  // An agent seen only in telemetry has no config to register, and the platform defaults an
  // empty one to nat-workflow-v1. It exists so its traces, evaluations and insights attach to a
  // real agent; deploying or chatting with it still needs a config.
  const createFromTraces = async () => {
    if (!tracedAgent) return;
    await createTracedAgent({ workspace, data: { name: tracedAgent, config: {} } }).catch(() => {
      // Rendered through errorText.
    });
  };

  // No fallback argument: getErrorMessage prefers one over a plain Error's own message.
  const failure = createError ?? tracedCreateError;
  const errorMessage =
    selectionError ??
    (failure ? getErrorMessage(failure as Error) || 'Failed to create agent' : undefined);

  const onUploadTab = tab === 'upload';
  const onTracesTab = tab === 'imported-traces';
  const onCreateTab = onUploadTab || onTracesTab;

  return (
    <FormModal
      open={open}
      onClose={resetAndClose}
      className="w-[720px] max-w-[90vw]"
      title="Instrument an agent with NeMo Platform"
      instruction="Integrated agents allow users to evaluate, optimize, and deploy agents."
      submitButtonText={replaceOrphan ? 'Replace and create' : 'Create'}
      onSubmit={(event) => {
        // The traced-agent choice is not part of the upload form, so it submits on its own
        // rather than through a resolver that would reject the empty name field.
        if (onTracesTab) {
          event.preventDefault();
          void createFromTraces();
          return;
        }
        handleSubmit(onSubmit)(event);
      }}
      disabled={isPending || isCreatingTraced}
      loading={isPending || isCreatingTraced}
      submitDisabled={onTracesTab ? !tracedAgent : entries.length === 0}
      errorText={onCreateTab ? errorMessage : undefined}
      slotFooterRight={
        onCreateTab ? undefined : (
          <Button color="brand" type="button" onClick={resetAndClose}>
            Close
          </Button>
        )
      }
    >
      <TabsRoot value={tab} onValueChange={(value) => setTab(value as NewAgentTab)}>
        <TabsList aria-label="Ways to instrument an agent">
          <TabsTrigger value="coding-agent-prompt">Coding agent prompt</TabsTrigger>
          <TabsTrigger value="upload">Upload agent</TabsTrigger>
          <TabsTrigger value="imported-traces">Create from traces</TabsTrigger>
        </TabsList>

        <TabsContent value="coding-agent-prompt" className="items-stretch p-0 pt-density-lg">
          <CodingAgentPromptEditor
            prompt={agentIntegrationPrompt({ workspace, baseUrl: PLATFORM_BASE_URL })}
          />
        </TabsContent>

        <TabsContent value="upload" className="items-stretch p-0 pt-density-lg">
          <Stack gap="density-md">
            <Text kind="label/semibold/md">Select agent config files</Text>
            <UploadRoot multiple disabled={isPending}>
              <UploadTrigger
                className="w-full"
                data-testid="agent-directory-dropzone"
                onDrop={onDirectoryDropped}
                slotAnchor={directoryName ? 'Choose a different directory' : 'Choose a directory'}
                slotHeaderText=" containing agent.yaml."
              >
                <UploadInputElement
                  ref={setDirectoryInput}
                  data-testid="agent-directory-input"
                  multiple
                  onChange={onDirectoryPicked}
                />
              </UploadTrigger>
            </UploadRoot>
            {entriesSummary ? <Text kind="body/regular/sm">{entriesSummary}</Text> : null}
            <ControlledTextInput
              useControllerProps={{ control, name: 'name' }}
              label="Name"
              formFieldProps={{ slotError: errors.name?.message }}
            />
          </Stack>
        </TabsContent>

        <TabsContent value="imported-traces" className="items-stretch p-0 pt-density-lg">
          {!tracedAgents.isLoading && tracedAgents.names.length === 0 ? (
            // The min-height gives the panel something to center within; the tab's content
            // is otherwise only as tall as this sentence.
            <Flex
              direction="col"
              align="center"
              justify="center"
              className="min-h-[220px] w-full text-center"
              data-testid="no-traced-agents"
            >
              <Text kind="body/regular/md" color="subtle" className="max-w-[64ch]">
                No traces with agent.name parameter found. You can import traces with the intake
                trace import skill to get started.
              </Text>
            </Flex>
          ) : (
            <Stack gap="density-sm">
              <Label>Unregistered agents that appear in ingested traces</Label>
              <Select
                aria-label="Agent from imported traces"
                value={tracedAgent}
                onValueChange={setTracedAgent}
                disabled={isCreatingTraced || tracedAgents.isLoading}
                placeholder={tracedAgents.isLoading ? 'Loading...' : 'Select an agent'}
                items={tracedAgents.names.map((name) => ({ value: name, children: name }))}
              />
            </Stack>
          )}
        </TabsContent>
      </TabsRoot>
    </FormModal>
  );
};
