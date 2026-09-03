// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@e2e-tests/utils/environment';
import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { useInsightsGetAnalysisConfig } from '@nemo/sdk/generated/insights/insights-analysis-configs';
import { getListSpansQueryKey } from '@nemo/sdk/generated/platform/spans';
import { getListTracesQueryKey } from '@nemo/sdk/generated/platform/traces';
import {
  AccordionContent,
  AccordionItem,
  AccordionRoot,
  AccordionTrigger,
  Button,
  Checkbox,
  Flex,
  FormField,
  Modal,
  Stack,
  TabsContent,
  TabsList,
  TabsRoot,
  TabsTrigger,
  Text,
  TextInput,
} from '@nvidia/foundations-react-core';
import {
  agentsFromTrajectories,
  type InsightsTriggerResult,
  isQualifiedModelRef,
  triggerInsightsRuns,
} from '@studio/api/insightsAnalysis';
import { queryClient } from '@studio/api/queryClient';
import { CodingAgentPromptEditor } from '@studio/components/CodingAgentPromptEditor';
import { ImportTracesResultList } from '@studio/components/ImportTracesModal/ImportTracesResultList';
import {
  DEFAULT_SPANS_SOURCE,
  ingestTraceFiles,
  readTraceFile,
  type SelectedTraceFile,
} from '@studio/components/ImportTracesModal/ingestTraceFiles';
import { InsightsModelPairFields } from '@studio/components/ImportTracesModal/InsightsModelPairFields';
import { parseAtifValue } from '@studio/components/ImportTracesModal/parseAtifTraces';
import { SelectedTraceFileTags } from '@studio/components/ImportTracesModal/SelectedTraceFileTags';
import type { ImportMethod, ImportTraceResult } from '@studio/components/ImportTracesModal/types';
import { traceImportPrompt } from '@studio/routes/agents/AgentDetailRoute/overview/codingAgentPrompts';
import { Upload } from 'lucide-react';
import { type ChangeEvent, type FC, useEffect, useMemo, useRef, useState } from 'react';

/** Formats whose records carry no agent field, so an agent-scoped import cannot reattribute them. */
const UNATTRIBUTABLE_FORMATS = new Set(['chat-completions', 'otlp-protobuf']);

const SOURCE_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]*$/;

export interface ImportTracesModalProps {
  open: boolean;
  onClose: () => void;
  workspace: string;
  /**
   * Attribute every imported record to this agent, overriding whatever agent the file
   * carries. Set when importing from an agent's own page; leave unset on the workspace-wide
   * Traces page to keep each file's own agent.
   */
  agent?: string;
}

/**
 * The two ways into Intake: hand the job to a coding agent running the `nemo-intake` skill —
 * which is what real trace volumes, formats, and live observability APIs need — or, for a small
 * set of files already on disk, upload them here directly.
 */
export const ImportTracesModal: FC<ImportTracesModalProps> = ({
  open,
  onClose,
  workspace,
  agent,
}) => {
  const [method, setMethod] = useState<ImportMethod>('skill');
  const [files, setFiles] = useState<SelectedTraceFile[]>([]);
  const [source, setSource] = useState(DEFAULT_SPANS_SOURCE);
  const [results, setResults] = useState<ImportTraceResult[] | null>(null);
  const [insightsResults, setInsightsResults] = useState<InsightsTriggerResult[] | null>(null);
  const [runInsights, setRunInsights] = useState(true);
  const [defaultModel, setDefaultModel] = useState('');
  const [fastModel, setFastModel] = useState('');
  const [isImporting, setIsImporting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const toast = useToast();

  /**
   * The agent whose stored pair can be shown up front: either the one this modal is
   * pinned to, or the single agent named by the chosen ATIF files. With several agents in
   * one import there is no single stored pair to display, so the fields stay empty
   * and any value entered applies to all of them.
   */
  const previewAgent = useMemo(() => {
    if (agent) return agent;
    const trajectories = files
      .filter(({ detection }) => detection.format === 'atif')
      .flatMap(
        ({ label, detection }) =>
          parseAtifValue(label, 'document' in detection ? detection.document : undefined).traces
      )
      .map(({ trajectory }) => trajectory);
    const named = agentsFromTrajectories(trajectories);
    return named.length === 1 ? named[0] : undefined;
  }, [agent, files]);

  const { data: analysisConfig, isError: configFailed } = useInsightsGetAnalysisConfig(
    workspace,
    previewAgent ?? '',
    { query: { enabled: runInsights && !!previewAgent, retry: false } }
  );

  /**
   * Seed the pickers with the stored pair so the modal shows what the run would use. Keyed on the
   * config's own identity, so re-selecting files for the same agent does not discard an edit — and
   * on `open`, because closing clears the pair while the cached config stays identical, which would
   * otherwise leave the pickers empty for the rest of the session.
   */
  useEffect(() => {
    setDefaultModel(analysisConfig?.default_model ?? '');
    setFastModel(analysisConfig?.fast_model ?? '');
  }, [open, analysisConfig?.id, analysisConfig?.default_model, analysisConfig?.fast_model]);

  const reset = () => {
    setMethod('skill');
    setFiles([]);
    setSource(DEFAULT_SPANS_SOURCE);
    setResults(null);
    setInsightsResults(null);
    setDefaultModel('');
    setFastModel('');
    setIsImporting(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleClose = () => {
    if (isImporting) return;
    reset();
    onClose();
  };

  /** Selections accumulate, so a set spread across directories can be picked up in passes. */
  const handleFilesSelected = async (event: ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(event.target.files ?? []);
    // Frees the input to re-fire for a file that was picked, removed, then picked again.
    event.target.value = '';
    if (selected.length === 0) return;

    const read = await Promise.all(selected.map(readTraceFile));
    setFiles((current) => {
      const known = new Set(current.map(({ id }) => id));
      return [...current, ...read.filter(({ id }) => !known.has(id))];
    });
    setResults(null);
    setInsightsResults(null);
  };

  const handleRemoveFile = (id: string) =>
    setFiles((current) => current.filter((file) => file.id !== id));

  const handleImport = async () => {
    setIsImporting(true);
    setInsightsResults(null);

    const { results: imported, agents } = await ingestTraceFiles(files, {
      workspace,
      agent,
      source: source.trim() || DEFAULT_SPANS_SOURCE,
    });
    setResults(imported);

    const succeeded = imported.filter(({ status }) => status === 'success').length;
    if (succeeded > 0) {
      void queryClient.invalidateQueries({ queryKey: getListTracesQueryKey(workspace) });
      void queryClient.invalidateQueries({ queryKey: getListSpansQueryKey(workspace) });
      toast.success(`Imported ${succeeded} file${succeeded === 1 ? '' : 's'}.`);
    }
    const failed = imported.length - succeeded;
    if (failed > 0) toast.error(`${failed} import${failed === 1 ? '' : 's'} failed.`);

    let insightsNeedsAttention = false;
    if (runInsights && agents.length > 0) {
      const triggered = await triggerInsightsRuns(workspace, agents, {
        default_model: defaultModel,
        fast_model: fastModel,
      });
      setInsightsResults(triggered);

      const started = triggered.filter(({ status }) => status === 'started').length;
      if (started > 0) {
        toast.success(`Queued ${started} insights analysis run${started === 1 ? '' : 's'}.`);
      }
      insightsNeedsAttention = started < triggered.length;
    }

    setIsImporting(false);

    // A clean import has nothing left to read — the toasts carry the counts. Anything that
    // failed, or an agent that was not enabled for analysis, keeps its reason on screen.
    if (failed === 0 && !insightsNeedsAttention) {
      reset();
      onClose();
    }
  };

  const importable = files.filter(({ detection }) => detection.format !== null);
  const needsSource = importable.some(({ detection }) => detection.format === 'spans');
  const unattributable =
    !!agent &&
    importable.some(({ detection }) => UNATTRIBUTABLE_FORMATS.has(detection.format as string));
  const invalidSource = needsSource && !SOURCE_PATTERN.test(source.trim());

  /** A malformed override would only fail once the analyze-job is already running. */
  const hasInvalidModelRef =
    runInsights &&
    [defaultModel, fastModel].some(
      (ref) => ref.trim().length > 0 && !isQualifiedModelRef(ref.trim())
    );

  return (
    <Modal
      open={open}
      onOpenChange={handleClose}
      slotHeading={
        <Stack gap="1">
          <Text kind="title/sm">Import traces</Text>
          <Text kind="body/regular/sm" className="text-secondary">
            Bring existing agent telemetry into Intake so it can be queried, analyzed, and turned
            into datasets.
          </Text>
        </Stack>
      }
      className="w-[90vw] max-w-[720px]"
      attributes={{ ModalFooter: { className: 'justify-end' } }}
      slotFooter={
        <Flex gap="density-sm">
          <Button kind="tertiary" onClick={handleClose} disabled={isImporting}>
            {results || method === 'skill' ? 'Close' : 'Cancel'}
          </Button>
          {method === 'files' && (
            <LoadingButton
              color="brand"
              onClick={handleImport}
              disabled={
                importable.length === 0 || isImporting || hasInvalidModelRef || invalidSource
              }
              loading={isImporting}
            >
              <Upload />
              {isImporting ? 'Importing...' : 'Import'}
            </LoadingButton>
          )}
        </Flex>
      }
    >
      <Stack gap="density-2xl">
        <TabsRoot value={method} onValueChange={(value) => setMethod(value as ImportMethod)}>
          <TabsList aria-label="Ways to import traces">
            <TabsTrigger value="skill" disabled={isImporting}>
              Use a skill
            </TabsTrigger>
            <TabsTrigger value="files" disabled={isImporting}>
              Select files
            </TabsTrigger>
          </TabsList>

          <TabsContent value="skill" className="items-stretch p-0 pt-density-lg">
            <CodingAgentPromptEditor
              prompt={traceImportPrompt({ workspace, agent, baseUrl: PLATFORM_BASE_URL })}
              className="h-80"
            />
          </TabsContent>

          <TabsContent value="files" className="items-stretch p-0 pt-density-lg">
            <Stack gap="density-2xl">
              <Text kind="body/regular/sm" color="secondary">
                Each file is read as ATIF, direct spans, captured chat completions, or an OTLP
                protobuf, and sent to the matching Intake endpoint for workspace{' '}
                <strong>{workspace}</strong>.
                {agent && (
                  <>
                    {' '}
                    Records that carry an agent are attributed to <strong>{agent}</strong>,
                    overriding the name in the file.
                  </>
                )}
              </Text>

              <FormField slotLabel="Trace files">
                <Stack gap="density-md">
                  <Flex gap="density-md" className="items-center">
                    <Button
                      kind="secondary"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={isImporting}
                    >
                      Upload files
                    </Button>
                    {files.length === 0 && (
                      <Text kind="body/regular/sm" color="secondary">
                        No files chosen
                      </Text>
                    )}
                  </Flex>
                  {files.length > 0 && (
                    <SelectedTraceFileTags
                      files={files}
                      onRemove={handleRemoveFile}
                      disabled={isImporting}
                    />
                  )}
                </Stack>
                <input
                  ref={fileInputRef}
                  data-testid="trace-files-input"
                  type="file"
                  accept=".json,.jsonl,.pb,.binpb,.protobuf,application/json"
                  multiple
                  hidden
                  onChange={handleFilesSelected}
                />
              </FormField>

              {unattributable && (
                <Text kind="body/regular/xs" color="secondary">
                  Chat completions and OTLP protobufs carry no agent field that Studio can rewrite,
                  so those records land in Intake but will not appear under <strong>{agent}</strong>{' '}
                  unless their own spans already name it.
                </Text>
              )}

              {needsSource && (
                <FormField
                  slotLabel="Span source"
                  slotError={
                    invalidSource
                      ? 'Use letters, digits, and . _ - only, starting with a letter or digit.'
                      : undefined
                  }
                >
                  <TextInput
                    value={source}
                    onValueChange={setSource}
                    placeholder={DEFAULT_SPANS_SOURCE}
                    aria-label="Span source"
                  />
                  <Text kind="body/regular/xs" color="secondary">
                    Recorded on direct-span batches to name where they came from, such as{' '}
                    <code>langsmith</code>. A file that names its own source keeps it.
                  </Text>
                </FormField>
              )}

              <AccordionRoot>
                <AccordionItem value="settings" className="border-b-0">
                  <AccordionTrigger chevronPosition="start">Advanced Options</AccordionTrigger>
                  <AccordionContent>
                    <Stack gap="density-xs" className="pt-density-md">
                      <Checkbox
                        checked={runInsights}
                        onChange={(event) => setRunInsights(event.target.checked)}
                        slotLabel="Run insights analysis after import"
                      />
                      <Text kind="body/regular/xs" color="secondary">
                        Queues one analyst run per agent named in the imported traces. The agent
                        must already be enabled with <code>nemo insights analysis enable</code>.
                      </Text>

                      {runInsights && (
                        <InsightsModelPairFields
                          workspace={workspace}
                          agent={previewAgent}
                          unresolved={!!previewAgent && configFailed}
                          defaultModel={defaultModel}
                          fastModel={fastModel}
                          onDefaultModelChange={setDefaultModel}
                          onFastModelChange={setFastModel}
                        />
                      )}
                    </Stack>
                  </AccordionContent>
                </AccordionItem>
              </AccordionRoot>

              {results && (
                <ImportTracesResultList results={results} insightsResults={insightsResults} />
              )}
            </Stack>
          </TabsContent>
        </TabsRoot>
      </Stack>
    </Modal>
  );
};
