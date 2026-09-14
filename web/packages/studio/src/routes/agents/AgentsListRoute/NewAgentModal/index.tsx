// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { getAgentsListAgentsQueryKey } from '@nemo/sdk/generated/agents/agents';
import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import {
  Button,
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
import { AgentSpecFilesetOrphanError } from '@studio/api/agents/agentSpecFileset';
import { useCreateAgentFromGitHub } from '@studio/api/agents/useCreateAgentFromGitHub';
import { useCreateAgentFromUpload } from '@studio/api/agents/useCreateAgentFromUpload';
import { CodingAgentPromptEditor } from '@studio/components/CodingAgentPromptEditor';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { agentIntegrationPrompt } from '@studio/routes/agents/AgentDetailRoute/overview/codingAgentPrompts';
import {
  AGENT_CONFIG_FILENAME,
  uploadAgentFormSchema,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import {
  type GitHubAgentSource,
  agentNameFromSource,
  parseGitHubSource,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/github';
import type {
  NewAgentModalProps,
  NewAgentTab,
  PickedFile,
  UploadAgentEntry,
  UploadAgentFormData,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/type';
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
import { CreateSecretModal } from '@studio/routes/SecretsListRoute/CreateSecretModal';
import { SecretSearchableSelect } from '@studio/routes/SecretsListRoute/SecretSearchableSelect';
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
  const [sourceLabel, setSourceLabel] = useState('');
  const [selectionError, setSelectionError] = useState<string | undefined>(undefined);
  const [replaceArmedFor, setReplaceArmedFor] = useState<string | null>(null);
  const [tab, setTab] = useState<NewAgentTab>('coding-agent-prompt');
  const [isSecretModalOpen, setSecretModalOpen] = useState(false);
  const [repoBlurred, setRepoBlurred] = useState(false);

  const onAgentCreated = (agent: Agent) => {
    toast.success(`Agent "${agent.name}" created`);
    void queryClient.invalidateQueries({ queryKey: getAgentsListAgentsQueryKey(workspace) });
    resetAndClose();
    if (agent.name) navigate(getAgentDetailRoute(workspace, agent.name));
  };

  const {
    mutateAsync: createAgent,
    error: createError,
    isPending: isUploading,
    reset: resetMutation,
  } = useCreateAgentFromUpload({ onSuccess: onAgentCreated });

  const {
    mutateAsync: createAgentFromRepo,
    error: repoError,
    isPending: isImporting,
    reset: resetRepoMutation,
  } = useCreateAgentFromGitHub({ onSuccess: onAgentCreated });

  const isPending = isUploading || isImporting;
  const onUploadTab = tab === 'upload';
  const onGitHubTab = tab === 'github';

  const {
    control,
    setValue,
    handleSubmit,
    reset: resetForm,
    formState: { errors },
  } = useForm({
    resolver: zodResolver(uploadAgentFormSchema),
    defaultValues: { name: '', repoUrl: '', secretKey: '' },
    disabled: isPending,
    mode: 'onChange',
  });

  // useWatch re-renders this modal on every keystroke; the summary depends only on entries.
  const entriesSummary = useMemo(
    () =>
      entries.length === 0
        ? undefined
        : `${sourceLabel} — ${entries.length} files, ${Math.max(1, Math.round(totalEntryBytes(entries) / 1000))} KB`,
    [sourceLabel, entries]
  );

  const watchedName = useWatch({ control, name: 'name' });
  const watchedRepoUrl = useWatch({ control, name: 'repoUrl' });
  const watchedSecretKey = useWatch({ control, name: 'secretKey' });

  // A repository is only a source once it parses; a half-typed URL must not enable submit.
  const parsedRepo = useMemo((): { source?: GitHubAgentSource; problem?: string } => {
    if (!watchedRepoUrl?.trim()) return {};
    try {
      return { source: parseGitHubSource(watchedRepoUrl) };
    } catch (error) {
      return { problem: getErrorMessage(error as Error) };
    }
  }, [watchedRepoUrl]);
  const repoSource = parsedRepo.source;
  // Held back until the field is left, so the message is not a running commentary on typing.
  const repoFieldError = errors.repoUrl?.message ?? (repoBlurred ? parsedRepo.problem : undefined);
  // Derived, not stored: an armed replace targets one fileset, so editing the name
  // disarms it in the same render rather than one render later.
  const replaceOrphan = replaceArmedFor !== null && replaceArmedFor === watchedName?.trim();

  const resetAndClose = () => {
    resetMutation();
    resetRepoMutation();
    resetForm({ name: '', repoUrl: '', secretKey: '' });
    setEntries([]);
    setSourceLabel('');
    setSelectionError(undefined);
    setReplaceArmedFor(null);
    setRepoBlurred(false);
    setTab('coding-agent-prompt');
    onClose();
  };

  // Directory reads finish out of order, so the newest selection has to win.
  const selectionSeq = useRef(0);
  const beginSelection = (): (() => boolean) => {
    const selection = ++selectionSeq.current;
    // Dropping the entries disables submit until this selection validates.
    resetMutation();
    resetRepoMutation();
    setEntries([]);
    setSelectionError(undefined);
    setReplaceArmedFor(null);
    return () => selection !== selectionSeq.current;
  };

  // Picks, drops and repositories all land here so every source is validated the same way.
  const acceptEntries = async (
    collected: UploadAgentEntry[],
    label: string,
    superseded: () => boolean
  ) => {
    setSourceLabel(label);

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

  const acceptPicked = (picked: PickedFile[], superseded: () => boolean) =>
    acceptEntries(
      collectAgentEntries(picked),
      picked[0]?.relativePath.split('/')[0] ?? '',
      superseded
    );

  const onRepoUrlBlur = () => {
    setRepoBlurred(true);
    if (!repoSource || watchedName?.trim()) return;
    setValue('name', agentNameFromSource(repoSource), { shouldValidate: true });
  };

  const rejectOversized = (count: number): boolean => {
    const oversized = tooManyPickedFiles(count);
    if (!oversized) return false;
    setEntries([]);
    setSourceLabel('');
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

  // Keyed on the active tab: the fileset has one source, and it is the one the user can see.
  const onSubmit: SubmitHandler<UploadAgentFormData> = async (formData) => {
    const name = formData.name.trim();
    try {
      if (onGitHubTab) {
        if (!repoSource) return;
        await createAgentFromRepo({
          workspace,
          name,
          source: repoSource,
          secretName: formData.secretKey?.trim() || undefined,
          replaceOrphanedFileset: replaceOrphan,
        });
        return;
      }
      await createAgent({ workspace, name, entries, replaceOrphanedFileset: replaceOrphan });
    } catch (error) {
      // An orphaned fileset is recoverable, so the next submit replaces it.
      setReplaceArmedFor(error instanceof AgentSpecFilesetOrphanError ? name : null);
    }
  };

  // No fallback argument: getErrorMessage prefers one over a plain Error's own message.
  const failure = onGitHubTab ? repoError : (selectionError ?? createError);
  const errorMessage =
    typeof failure === 'string'
      ? failure
      : failure
        ? getErrorMessage(failure) || 'Failed to create agent'
        : undefined;

  return (
    <>
      <FormModal
        open={open}
        onClose={resetAndClose}
        className="w-[720px] max-w-[90vw]"
        title="Instrument an agent with NeMo Platform"
        instruction="Integrated agents allow users to evaluate, optimize, and deploy agents."
        submitButtonText={replaceOrphan ? 'Replace and create' : 'Create'}
        onSubmit={handleSubmit(onSubmit)}
        disabled={isPending}
        loading={isPending}
        submitDisabled={onGitHubTab ? !repoSource : entries.length === 0}
        errorText={onUploadTab || onGitHubTab ? errorMessage : undefined}
        slotFooterRight={
          onUploadTab || onGitHubTab ? undefined : (
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
            <TabsTrigger value="github">GitHub repository</TabsTrigger>
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
                  slotAnchor={sourceLabel ? 'Choose a different directory' : 'Choose a directory'}
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

          <TabsContent value="github" className="items-stretch p-0 pt-density-lg">
            <Stack gap="density-md">
              <ControlledTextInput
                label="Repository"
                disabled={isPending}
                useControllerProps={{ control, name: 'repoUrl' }}
                formFieldProps={{
                  slotInfo:
                    'github.com/owner/repo, optionally with @branch and #sub/directory. The files are read from GitHub on demand, not copied.',
                  slotError: repoFieldError,
                  // FormField drops slotError unless the field is also marked failed.
                  status: repoFieldError ? 'error' : undefined,
                }}
                attributes={{ Input: { onBlur: onRepoUrlBlur } }}
              />
              <SecretSearchableSelect
                workspace={workspace}
                queryEnabled={open && onGitHubTab && Boolean(workspace)}
                ensureOptionValue={watchedSecretKey || undefined}
                useControllerProps={{ control, name: 'secretKey' }}
                onRequestNewSecret={() => setSecretModalOpen(true)}
                triggerPlaceholder=""
                formFieldProps={{
                  slotLabel: 'Access token secret',
                  slotInfo:
                    'Required for a private repository. The token stays in the platform and is never sent to your browser.',
                  slotError: errors.secretKey?.message,
                }}
              />
              <ControlledTextInput
                useControllerProps={{ control, name: 'name' }}
                label="Name"
                formFieldProps={{ slotError: errors.name?.message }}
              />
            </Stack>
          </TabsContent>
        </TabsRoot>
      </FormModal>
      {isSecretModalOpen ? (
        <CreateSecretModal
          workspace={workspace}
          open
          onClose={() => setSecretModalOpen(false)}
          onSecretCreated={(secretName) => {
            setValue('secretKey', secretName, { shouldValidate: true });
            setSecretModalOpen(false);
          }}
        />
      ) : null}
    </>
  );
};
