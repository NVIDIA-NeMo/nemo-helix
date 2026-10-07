// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { ControlledCheckbox } from '@nemo/common/src/components/form/ControlledCheckbox';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getAgentsListDeploymentsQueryKey,
  useAgentsCreateDeployment,
} from '@nemo/sdk/generated/agents/agent-deployments';
import {
  getAgentsListAgentsQueryKey,
  useAgentsCreateAgent,
} from '@nemo/sdk/generated/agents/agents';
import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
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
import { AgentSpecFilesetOrphanError } from '@studio/api/agents/agentSpecFileset';
import { buildThenDeployNavigation } from '@studio/api/agents/buildThenDeploy';
import { useCreateAgentFromGitHub } from '@studio/api/agents/useCreateAgentFromGitHub';
import { useCreateAgentFromUpload } from '@studio/api/agents/useCreateAgentFromUpload';
import {
  type DeploymentMode,
  IMAGE_DEPLOYMENT_MODES,
  useDeploymentModes,
} from '@studio/api/agents/useDeploymentModes';
import { useImageBuildsUnsupported } from '@studio/api/agents/useImageBuildsUnsupported';
import { CodingAgentPromptEditor } from '@studio/components/CodingAgentPromptEditor';
import { DeploymentModeSelect } from '@studio/components/DeploymentModeSelect';
import { ImageBuildFirstNotice } from '@studio/components/ImageBuildFirstNotice';
import {
  AGENT_CONTAINER_DEPLOYMENTS_ENABLED,
  PLATFORM_BASE_URL,
} from '@studio/constants/environment';
import { agentIntegrationPrompt } from '@studio/routes/agents/AgentDetailRoute/overview/codingAgentPrompts';
import {
  AGENT_CONFIG_FILENAME,
  UPLOAD_AGENT_FORM_DEFAULTS,
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
import { useTraceAgentNames } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/useTraceAgentNames';
import {
  agentNameFromConfig,
  collectAgentEntries,
  findNonUtf8Path,
  parseAgentConfig,
  pickedFromDataTransfer,
  pickedFromFileList,
  selectionRootName,
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
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { type SubmitHandler, useForm, useWatch } from 'react-hook-form';
import { useNavigate } from 'react-router';

const OFFERED_ON_CREATE: readonly DeploymentMode[] = AGENT_CONTAINER_DEPLOYMENTS_ENABLED
  ? ['subprocess', ...IMAGE_DEPLOYMENT_MODES]
  : ['subprocess'];

export const NewAgentModal: FC<NewAgentModalProps> = ({
  open,
  onClose,
  workspace,
  initialName,
}) => {
  const toast = useToast();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const filesInputRef = useRef<HTMLInputElement>(null);
  const setFolderInput = useCallback((node: HTMLInputElement | null) => {
    // webkitdirectory is absent from React's input attribute types.
    node?.setAttribute('webkitdirectory', '');
  }, []);
  const [entries, setEntries] = useState<UploadAgentEntry[]>([]);
  const [sourceLabel, setSourceLabel] = useState('');
  const [selectionError, setSelectionError] = useState<string | undefined>(undefined);
  const [replaceArmedFor, setReplaceArmedFor] = useState<string | null>(null);
  const [tab, setTab] = useState<NewAgentTab>('upload');
  const [isSecretModalOpen, setSecretModalOpen] = useState(false);
  const [repoBlurred, setRepoBlurred] = useState(false);
  const [tracedAgent, setTracedAgent] = useState('');
  // Set on submit, so an agent created from traces, which has no config to run, is never deployed.
  const deployAfterCreate = useRef<{ mode: DeploymentMode; buildImage: boolean } | null>(null);

  // Toasts live on the hook, not on mutate(): navigating to the new agent unmounts this modal.
  const { mutate: deployAgent } = useAgentsCreateDeployment({
    mutation: {
      onSuccess: (deployment) => {
        toast.success(`Deploying agent "${deployment.agent}"`);
        void queryClient.invalidateQueries({
          queryKey: getAgentsListDeploymentsQueryKey(workspace),
        });
      },
      onError: (error, { data }) => {
        toast.error(
          `Agent "${data.agent}" was created, but deploying it failed: ${getErrorMessage(error) || 'unknown error'}`
        );
      },
    },
  });

  const onAgentCreated = (agent: Agent) => {
    toast.success(`Agent "${agent.name}" created`);
    void queryClient.invalidateQueries({ queryKey: getAgentsListAgentsQueryKey(workspace) });
    const deployment = deployAfterCreate.current;
    resetAndClose();
    if (!agent.name) return;
    if (deployment?.buildImage) {
      // The build outlives this modal, so the agent's page runs it and deploys the tag.
      navigate(...buildThenDeployNavigation(workspace, agent.name, { mode: deployment.mode }));
      return;
    }
    if (deployment) {
      deployAgent({ workspace, data: { agent: agent.name, deployment_mode: deployment.mode } });
    }
    navigate(getAgentDetailRoute(workspace, agent.name));
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
  const onTracesTab = tab === 'imported-traces';
  const onCreateTab = onUploadTab || onGitHubTab || onTracesTab;

  const {
    control,
    setValue,
    handleSubmit,
    reset: resetForm,
    formState: { errors },
  } = useForm({
    resolver: zodResolver(uploadAgentFormSchema),
    defaultValues: { ...UPLOAD_AGENT_FORM_DEFAULTS, name: initialName ?? '' },
    disabled: isPending,
    mode: 'onChange',
  });

  const tracedAgents = useTraceAgentNames(workspace, open && tab === 'imported-traces');

  // A new agent has no image yet, so a mode that needs one is offered only if the platform can build it.
  const deploymentModeState = useDeploymentModes(workspace, { enabled: open });
  const imageBuildsUnsupported = useImageBuildsUnsupported();
  const deploymentModes = useMemo<readonly DeploymentMode[]>(
    () =>
      deploymentModeState.status === 'ready'
        ? OFFERED_ON_CREATE.filter(
            (mode) =>
              deploymentModeState.withoutImage.includes(mode) ||
              (deploymentModeState.enabled.includes(mode) && !imageBuildsUnsupported)
          )
        : [],
    [deploymentModeState, imageBuildsUnsupported]
  );
  const modeNeedsImageBuild = (mode: DeploymentMode) =>
    deploymentModeState.status === 'ready' && !deploymentModeState.withoutImage.includes(mode);
  const isModesLoading = deploymentModeState.status === 'loading';
  const canDeployOnCreate = deploymentModes.length > 0;

  const {
    mutateAsync: createTracedAgent,
    error: tracedCreateError,
    isPending: isCreatingTraced,
    reset: resetTracedMutation,
  } = useAgentsCreateAgent({ mutation: { onSuccess: onAgentCreated } });

  // useWatch re-renders this modal on every keystroke; the summary depends only on entries.
  const entriesSummary = useMemo(() => {
    if (entries.length === 0) return undefined;
    const size = `${entries.length} ${entries.length === 1 ? 'file' : 'files'}, ${Math.max(1, Math.round(totalEntryBytes(entries) / 1000))} KB`;
    return sourceLabel ? `${sourceLabel} — ${size}` : size;
  }, [sourceLabel, entries]);

  const watchedName = useWatch({ control, name: 'name' });
  const watchedRepoUrl = useWatch({ control, name: 'repoUrl' });
  const watchedSecretKey = useWatch({ control, name: 'secretKey' });
  const watchedDeploy = useWatch({ control, name: 'deploy' });
  const watchedDeploymentMode = useWatch({ control, name: 'deploymentMode' });

  useEffect(() => {
    const firstMode = deploymentModes.at(0);
    if (firstMode && !deploymentModes.includes(watchedDeploymentMode)) {
      setValue('deploymentMode', firstMode);
    }
  }, [deploymentModes, watchedDeploymentMode, setValue]);

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
    resetTracedMutation();
    resetForm({ ...UPLOAD_AGENT_FORM_DEFAULTS, name: initialName ?? '' });
    deployAfterCreate.current = null;
    setEntries([]);
    setSourceLabel('');
    setSelectionError(undefined);
    setReplaceArmedFor(null);
    setRepoBlurred(false);
    setTab('upload');
    setTracedAgent('');
    onClose();
  };

  // Selection reads finish out of order, so the newest selection has to win.
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
      setValue('name', initialName ?? agentNameFromConfig(config) ?? '', {
        shouldValidate: true,
      });
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
    acceptEntries(collectAgentEntries(picked), selectionRootName(picked), superseded);

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

  const onFilesPicked: ChangeEventHandler<HTMLInputElement> = async (event) => {
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

  const onFilesDropped: DragEventHandler<HTMLLabelElement> = async (event) => {
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
    deployAfterCreate.current =
      formData.deploy && canDeployOnCreate
        ? {
            mode: formData.deploymentMode,
            buildImage: modeNeedsImageBuild(formData.deploymentMode),
          }
        : null;
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

  // An agent seen only in telemetry has no config to register, and the platform defaults an
  // empty one to nat-workflow-v1. It exists so its traces, evaluations and insights attach to a
  // real agent; deploying or chatting with it still needs a config.
  const createFromTraces = async () => {
    if (!tracedAgent) return;
    deployAfterCreate.current = null;
    await createTracedAgent({ workspace, data: { name: tracedAgent, config: {} } }).catch(() => {
      // Rendered through errorText.
    });
  };

  // No fallback argument: getErrorMessage prefers one over a plain Error's own message.
  const failure = onGitHubTab
    ? repoError
    : onTracesTab
      ? tracedCreateError
      : (selectionError ?? createError);
  const errorMessage =
    typeof failure === 'string'
      ? failure
      : failure
        ? getErrorMessage(failure) || 'Failed to create agent'
        : undefined;

  const busy = isPending || isCreatingTraced;
  const awaitingDeployModes = Boolean(watchedDeploy) && isModesLoading;

  const deployFields =
    canDeployOnCreate || isModesLoading ? (
      <Stack gap="density-md">
        <ControlledCheckbox
          useControllerProps={{ control, name: 'deploy' }}
          slotLabel="Deploy after creating"
        />
        {watchedDeploy && deploymentModes.length > 1 ? (
          <DeploymentModeSelect
            useControllerProps={{ control, name: 'deploymentMode' }}
            modes={deploymentModes}
            loading={isModesLoading}
          />
        ) : null}
        {watchedDeploy && modeNeedsImageBuild(watchedDeploymentMode) ? (
          <ImageBuildFirstNotice mode={watchedDeploymentMode} />
        ) : null}
      </Stack>
    ) : null;

  return (
    <>
      <FormModal
        open={open}
        onClose={resetAndClose}
        className="w-[800px] max-w-[90vw]"
        title="Register an agent with NeMo Helix"
        submitButtonText={replaceOrphan ? 'Replace and register' : 'Register'}
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
        disabled={busy}
        loading={busy}
        submitDisabled={
          onGitHubTab
            ? !repoSource || awaitingDeployModes
            : onTracesTab
              ? !tracedAgent
              : entries.length === 0 || awaitingDeployModes
        }
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
          <TabsList aria-label="Ways to register an agent">
            <TabsTrigger value="upload">Register with code upload</TabsTrigger>
            <TabsTrigger value="github">Register from GitHub</TabsTrigger>
            <TabsTrigger value="imported-traces">Register from traces</TabsTrigger>
            <TabsTrigger value="coding-agent-prompt">Coding agent prompt</TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="items-stretch p-0 pt-density-lg">
            <Stack gap="density-md">
              <Text kind="label/semibold/md">Select agent config files</Text>
              <UploadRoot multiple disabled={isPending}>
                <UploadTrigger
                  className="w-full"
                  data-testid="agent-directory-dropzone"
                  onDrop={onFilesDropped}
                  slotAnchor={sourceLabel ? 'Choose a different folder' : 'Choose a folder'}
                  slotHeaderText=" containing agent.yaml, or drop it here."
                >
                  <UploadInputElement
                    ref={setFolderInput}
                    data-testid="agent-directory-input"
                    multiple
                    onChange={onFilesPicked}
                  />
                </UploadTrigger>
              </UploadRoot>
              <Flex justify="end">
                {/* A native picker offers files or directories, never both, so the file pick is its own control. */}
                <Button
                  kind="tertiary"
                  size="small"
                  type="button"
                  disabled={isPending}
                  onClick={() => filesInputRef.current?.click()}
                >
                  Choose files instead
                </Button>
                <input
                  ref={filesInputRef}
                  data-testid="agent-files-input"
                  type="file"
                  multiple
                  className="hidden"
                  onChange={onFilesPicked}
                />
              </Flex>
              {entriesSummary ? <Text kind="body/regular/sm">{entriesSummary}</Text> : null}
              <ControlledTextInput
                useControllerProps={{ control, name: 'name' }}
                label="Name"
                formFieldProps={{ slotError: errors.name?.message }}
              />
              {deployFields}
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
              {deployFields}
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
                  onValueChange={(value) => {
                    resetTracedMutation();
                    setTracedAgent(value);
                  }}
                  disabled={isCreatingTraced || tracedAgents.isLoading}
                  placeholder={tracedAgents.isLoading ? 'Loading...' : 'Select an agent'}
                  items={tracedAgents.names.map((name) => ({ value: name, children: name }))}
                />
              </Stack>
            )}
          </TabsContent>

          <TabsContent value="coding-agent-prompt" className="items-stretch p-0 pt-density-lg">
            <CodingAgentPromptEditor
              prompt={agentIntegrationPrompt({ workspace, baseUrl: PLATFORM_BASE_URL })}
            />
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
