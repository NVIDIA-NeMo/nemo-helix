// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { RadioCard } from '@nemo/common/src/components/RadioCard';
import { getEntityReference, getPartsFromReference } from '@nemo/common/src/namedEntity';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getAgentOptimizationListRunStrategyJobsQueryKey,
  useAgentOptimizationListStrategies,
} from '@nemo/sdk/generated/agent-optimization/agent-optimization';
import type { RunStrategyJob } from '@nemo/sdk/generated/agent-optimization/schema/RunStrategyJob';
import { useFilesListFilesetFiles, useFilesListFilesets } from '@nemo/sdk/generated/platform/files';
import {
  Banner,
  Button,
  Flex,
  RadioGroupRoot,
  Select,
  Stack,
  Text,
  UploadInputElement,
  UploadRoot,
  UploadTrigger,
} from '@nvidia/foundations-react-core';
import {
  isYamlPath,
  optimizeBundleProblems,
  parseOptimizeConfig,
} from '@studio/api/agents/optimizeBundle';
import { useLaunchOptimizeStudy } from '@studio/api/agents/useLaunchOptimizeStudy';
import type { FilesetEntry } from '@studio/api/files/uploadFilesetEntries';
import { MAX_PICKED_FILES } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import type { PickedFile } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/type';
import {
  collectAgentEntries,
  pickedFromDataTransfer,
  pickedFromFileList,
  selectionRootName,
  totalEntryBytes,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';
import { getOptimizeJobRoute } from '@studio/routes/utils';
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
import { useNavigate } from 'react-router';

interface LaunchOptimizeModalProps {
  open: boolean;
  onClose: () => void;
  workspace: string;
  agentName: string;
}

interface Bundle {
  label: string;
  entries: FilesetEntry[];
  /** Every YAML in the bundle, parsed where it is a mapping. */
  configs: Record<string, Record<string, unknown> | undefined>;
}

/**
 * The `legacy` strategy's spec requires a config, and it is the only config whose shape Studio can
 * preflight in the browser.
 */
const PREFLIGHT_STRATEGY = 'legacy';

const NO_BUNDLE = '__none__';
const UPLOAD_BUNDLE = '__upload__';

const readBundle = async (picked: PickedFile[]): Promise<Bundle> => {
  const entries = collectAgentEntries(picked);
  if (entries.length === 0) throw new Error('That selection has no uploadable files.');

  const parsed = await Promise.all(
    entries
      .filter((entry) => isYamlPath(entry.path))
      .map(async (entry) => [entry.path, parseOptimizeConfig(await entry.file.text())] as const)
  );
  if (parsed.length === 0) {
    throw new Error('No config in that selection: expected at least one YAML file.');
  }

  return { label: selectionRootName(picked), entries, configs: Object.fromEntries(parsed) };
};

export const LaunchOptimizeModal: FC<LaunchOptimizeModalProps> = ({
  open,
  onClose,
  workspace,
  agentName,
}) => {
  const toast = useToast();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [strategy, setStrategy] = useState('');
  const [source, setSource] = useState(NO_BUNDLE);
  const [bundle, setBundle] = useState<Bundle | undefined>();
  const [configPath, setConfigPath] = useState('');
  const [selectionError, setSelectionError] = useState<string | undefined>();

  const strategies = useAgentOptimizationListStrategies();
  const filesets = useFilesListFilesets(workspace, { page_size: 100, sort: '-created_at' });

  const selectedFileset =
    source === NO_BUNDLE || source === UPLOAD_BUNDLE ? undefined : getPartsFromReference(source);
  const filesetFiles = useFilesListFilesetFiles(
    selectedFileset?.workspace ?? '',
    selectedFileset?.name ?? '',
    undefined,
    { query: { enabled: !!selectedFileset } }
  );

  const filesInput = useRef<HTMLInputElement>(null);
  const setFolderInput = useCallback((node: HTMLInputElement | null) => {
    // webkitdirectory is absent from React's input attribute types.
    node?.setAttribute('webkitdirectory', '');
  }, []);

  const {
    mutate: launch,
    error: launchError,
    isPending,
    reset: resetLaunch,
  } = useLaunchOptimizeStudy({
    onSuccess: (job: RunStrategyJob) => {
      toast.success(`Optimization study "${job.name}" submitted`);
      void queryClient.invalidateQueries({
        queryKey: getAgentOptimizationListRunStrategyJobsQueryKey(workspace),
      });
      onClose();
      if (job.name) navigate(getOptimizeJobRoute(workspace, job.name, job.spec?.strategy));
    },
  });

  const configPaths = useMemo(() => {
    if (source === UPLOAD_BUNDLE) return Object.keys(bundle?.configs ?? {}).sort();
    return (filesetFiles.data?.data ?? [])
      .map((file) => file.path)
      .filter(isYamlPath)
      .sort();
  }, [source, bundle, filesetFiles.data]);

  const problems = useMemo(() => {
    if (strategy !== PREFLIGHT_STRATEGY || source !== UPLOAD_BUNDLE || !bundle || !configPath) {
      return [];
    }
    const config = bundle.configs[configPath];
    if (!config) return [`${configPath} is not valid YAML with a mapping at the top level.`];
    return optimizeBundleProblems({
      config,
      bundlePaths: new Set(bundle.entries.map((entry) => entry.path)),
      agent: agentName,
    });
  }, [strategy, source, bundle, configPath, agentName]);

  const summary = useMemo(() => {
    if (!bundle) return undefined;
    const count = bundle.entries.length;
    const size = `${count} ${count === 1 ? 'file' : 'files'}, ${Math.max(1, Math.round(totalEntryBytes(bundle.entries) / 1000))} KB`;
    return bundle.label ? `${bundle.label} — ${size}` : size;
  }, [bundle]);

  const resetBundle = () => {
    resetLaunch();
    setBundle(undefined);
    setConfigPath('');
    setSelectionError(undefined);
  };

  // Selection reads finish out of order, so the newest selection has to win.
  const selectionSeq = useRef(0);
  const acceptPicked = async (loadPicked: () => Promise<PickedFile[]> | PickedFile[]) => {
    const selection = ++selectionSeq.current;
    resetBundle();

    try {
      const picked = await loadPicked();
      if (picked.length > MAX_PICKED_FILES) {
        throw new Error(
          `That selection holds ${picked.length.toLocaleString()} files; an optimize bundle should hold only the config and the assets it references.`
        );
      }
      const next = await readBundle(picked);
      if (selection !== selectionSeq.current) return;
      const paths = Object.keys(next.configs);
      setBundle(next);
      setConfigPath(paths.length === 1 ? (paths[0] ?? '') : '');
    } catch (error) {
      if (selection !== selectionSeq.current) return;
      setSelectionError(getErrorMessage(error as Error) || 'Could not read that selection.');
    }
  };

  const onFilesPicked: ChangeEventHandler<HTMLInputElement> = (event) => {
    const files = Array.from(event.target.files ?? []);
    event.target.value = '';
    if (files.length === 0) return;
    void acceptPicked(() => pickedFromFileList(files));
  };

  const onFilesDropped: DragEventHandler<HTMLLabelElement> = (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (isPending) return;
    const items = Array.from(event.dataTransfer.items);
    if (items.length === 0) return;
    void acceptPicked(() => pickedFromDataTransfer(items));
  };

  const onSourceChange = (next: string) => {
    selectionSeq.current += 1;
    resetBundle();
    setSource(next);
  };

  const close = () => {
    selectionSeq.current += 1;
    resetBundle();
    setStrategy('');
    setSource(NO_BUNDLE);
    onClose();
  };

  const errorText =
    selectionError ??
    (launchError ? getErrorMessage(launchError) || 'Failed to submit the study' : undefined);

  const filesetItems = (filesets.data?.data ?? []).map((fileset) => ({
    value: getEntityReference(fileset),
    children: fileset.name,
  }));

  const needsConfig = source !== NO_BUNDLE;
  const missingConfig = strategy === PREFLIGHT_STRATEGY && source === NO_BUNDLE;

  return (
    <FormModal
      open={open}
      onClose={close}
      className="w-[720px] max-w-[90vw]"
      title="Optimize agent"
      instruction="Choose how to optimize this agent. Some strategies take a bundle of files or a single config YAML; the strategy reports what it is missing when the study is submitted."
      submitButtonText="Run strategy"
      onSubmit={(event) => {
        event.preventDefault();
        if (!strategy) return;
        if (source === NO_BUNDLE) {
          launch({ workspace, agentName, strategy });
        } else if (source === UPLOAD_BUNDLE) {
          if (!bundle) return;
          launch({
            workspace,
            agentName,
            strategy,
            bundle: { kind: 'upload', entries: bundle.entries, optimizeConfig: configPath },
          });
        } else {
          launch({
            workspace,
            agentName,
            strategy,
            bundle: { kind: 'fileset', fileset: source, optimizeConfig: configPath },
          });
        }
      }}
      disabled={isPending}
      loading={isPending}
      submitDisabled={
        !strategy || missingConfig || (needsConfig && !configPath) || problems.length > 0
      }
      errorText={errorText}
    >
      <Stack gap="density-md">
        <Text kind="label/semibold/md">Strategy</Text>
        {strategies.error ? (
          <Banner kind="inline" status="error">
            {getErrorMessage(strategies.error) || 'Could not load optimization strategies.'}
          </Banner>
        ) : strategies.isPending ? (
          <Text kind="body/regular/sm">Loading strategies…</Text>
        ) : (
          <RadioGroupRoot
            name="strategy"
            aria-label="Strategy"
            value={strategy}
            onValueChange={setStrategy}
            disabled={isPending}
            className="w-full"
          >
            <Stack gap="density-sm">
              {(strategies.data?.data ?? []).map((item) => (
                <RadioCard
                  key={item.name}
                  value={item.name}
                  compact
                  labelKind="body/bold/md"
                  descriptionKind="body/regular/sm"
                  label={item.name.toUpperCase()}
                  description={item.description || undefined}
                />
              ))}
            </Stack>
          </RadioGroupRoot>
        )}

        <Text kind="label/semibold/md">Configuration (optional)</Text>
        <Select
          aria-label="Configuration source"
          value={source}
          onValueChange={onSourceChange}
          disabled={isPending}
          items={[
            { value: NO_BUNDLE, children: 'None — the agent only' },
            { value: UPLOAD_BUNDLE, children: 'Upload files' },
            ...filesetItems,
          ]}
        />

        {source === UPLOAD_BUNDLE ? (
          <>
            <UploadRoot multiple disabled={isPending}>
              <UploadTrigger
                className="w-full"
                data-testid="optimize-bundle-dropzone"
                onDrop={onFilesDropped}
                slotAnchor={bundle ? 'Choose a different folder' : 'Choose a folder'}
                slotHeaderText=" holding the config and the files it references, or drop files here."
              >
                <UploadInputElement
                  ref={setFolderInput}
                  data-testid="optimize-bundle-input"
                  multiple
                  onChange={onFilesPicked}
                />
              </UploadTrigger>
            </UploadRoot>
            <Flex align="center" gap="density-xs">
              <Text kind="body/regular/sm">
                Or pick individual files, such as a single agent.yaml:
              </Text>
              <Button
                kind="tertiary"
                size="small"
                disabled={isPending}
                onClick={() => filesInput.current?.click()}
              >
                Choose files
              </Button>
              <input
                ref={filesInput}
                type="file"
                multiple
                className="hidden"
                data-testid="optimize-files-input"
                onChange={onFilesPicked}
              />
            </Flex>
            {summary ? <Text kind="body/regular/sm">{summary}</Text> : null}
          </>
        ) : null}

        {needsConfig && (bundle || selectedFileset) ? (
          <Select
            aria-label="Config file"
            value={configPath}
            onValueChange={setConfigPath}
            disabled={isPending || filesetFiles.isFetching}
            placeholder={
              filesetFiles.isError
                ? 'Could not load the files in this fileset'
                : !filesetFiles.isFetching && configPaths.length === 0
                  ? 'No YAML files in this fileset'
                  : 'Select the config YAML'
            }
            items={configPaths.map((path) => ({ value: path, children: path }))}
          />
        ) : null}

        {problems.length > 0 ? (
          <Banner status="error" kind="inline" data-testid="optimize-bundle-problems">
            <Stack gap="density-xs">
              <Text kind="body/semibold/sm">
                {`${problems.length} ${problems.length === 1 ? 'problem' : 'problems'} in ${configPath}`}
              </Text>
              <ul className="list-disc pl-5">
                {problems.map((problem) => (
                  <li key={problem}>
                    <Text kind="body/regular/sm">{problem}</Text>
                  </li>
                ))}
              </ul>
            </Stack>
          </Banner>
        ) : null}
      </Stack>
    </FormModal>
  );
};
