// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { getAgentsListOptimizeJobsQueryKey } from '@nemo/sdk/generated/agents/agents';
import type { OptimizeJob } from '@nemo/sdk/generated/agents/schema/OptimizeJob';
import {
  Banner,
  Select,
  Stack,
  Text,
  UploadInputElement,
  UploadRoot,
  UploadTrigger,
} from '@nvidia/foundations-react-core';
import {
  isYamlPath,
  looksLikeOptimizeConfig,
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
import { getAgentOptimizationDetailRoute } from '@studio/routes/utils';
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
  configs: Record<string, Record<string, unknown>>;
}

const readBundle = async (picked: PickedFile[]): Promise<Bundle> => {
  const entries = collectAgentEntries(picked);
  if (entries.length === 0) throw new Error('That selection has no uploadable files.');

  const parsed = await Promise.all(
    entries
      .filter((entry) => isYamlPath(entry.path))
      .map(async (entry) => [entry.path, parseOptimizeConfig(await entry.file.text())] as const)
  );
  const configs = Object.fromEntries(
    parsed.filter(([, config]) => looksLikeOptimizeConfig(config))
  ) as Bundle['configs'];
  if (Object.keys(configs).length === 0) {
    throw new Error(
      'No optimize config in that selection: expected a YAML file with an optimizer section.'
    );
  }

  return { label: selectionRootName(picked), entries, configs };
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

  const [bundle, setBundle] = useState<Bundle | undefined>();
  const [configPath, setConfigPath] = useState('');
  const [selectionError, setSelectionError] = useState<string | undefined>();

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
    onSuccess: (job: OptimizeJob) => {
      toast.success(`Optimization study "${job.name}" submitted`);
      void queryClient.invalidateQueries({
        queryKey: getAgentsListOptimizeJobsQueryKey(workspace),
      });
      onClose();
      if (job.name) navigate(getAgentOptimizationDetailRoute(workspace, job.name));
    },
  });

  const configPaths = useMemo(() => Object.keys(bundle?.configs ?? {}).sort(), [bundle]);

  const problems = useMemo(() => {
    const config = bundle?.configs[configPath];
    if (!bundle || !config) return [];
    return optimizeBundleProblems({
      config,
      bundlePaths: new Set(bundle.entries.map((entry) => entry.path)),
      agent: agentName,
    });
  }, [bundle, configPath, agentName]);

  const summary = useMemo(() => {
    if (!bundle) return undefined;
    const count = bundle.entries.length;
    const size = `${count} ${count === 1 ? 'file' : 'files'}, ${Math.max(1, Math.round(totalEntryBytes(bundle.entries) / 1000))} KB`;
    return bundle.label ? `${bundle.label} — ${size}` : size;
  }, [bundle]);

  // Selection reads finish out of order, so the newest selection has to win.
  const selectionSeq = useRef(0);
  const acceptPicked = async (loadPicked: () => Promise<PickedFile[]> | PickedFile[]) => {
    const selection = ++selectionSeq.current;
    resetLaunch();
    setBundle(undefined);
    setConfigPath('');
    setSelectionError(undefined);

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

  const close = () => {
    selectionSeq.current += 1;
    resetLaunch();
    setBundle(undefined);
    setConfigPath('');
    setSelectionError(undefined);
    onClose();
  };

  const errorText =
    selectionError ??
    (launchError ? getErrorMessage(launchError) || 'Failed to submit the study' : undefined);

  return (
    <FormModal
      open={open}
      onClose={close}
      className="w-[720px] max-w-[90vw]"
      title="Optimize agent"
      instruction="Upload an optimize bundle: the optimize config plus the dataset and any other files it references. Each trial runs this agent with one set of parameters from the config's search space."
      submitButtonText="Start study"
      onSubmit={(event) => {
        event.preventDefault();
        if (!bundle || !configPath) return;
        launch({ workspace, agentName, entries: bundle.entries, optimizeConfig: configPath });
      }}
      disabled={isPending}
      loading={isPending}
      submitDisabled={!bundle || !configPath || problems.length > 0}
      errorText={errorText}
    >
      <Stack gap="density-md">
        <Text kind="label/semibold/md">Optimize bundle</Text>
        <UploadRoot multiple disabled={isPending}>
          <UploadTrigger
            className="w-full"
            data-testid="optimize-bundle-dropzone"
            onDrop={onFilesDropped}
            slotAnchor={bundle ? 'Choose a different folder' : 'Choose a folder'}
            slotHeaderText=" containing the optimize config, or drop it here."
          >
            <UploadInputElement
              ref={setFolderInput}
              data-testid="optimize-bundle-input"
              multiple
              onChange={onFilesPicked}
            />
          </UploadTrigger>
        </UploadRoot>
        {summary ? <Text kind="body/regular/sm">{summary}</Text> : null}
        {bundle ? (
          <Select
            aria-label="Optimize config"
            value={configPath}
            onValueChange={setConfigPath}
            disabled={isPending}
            placeholder="Select the optimize config"
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
