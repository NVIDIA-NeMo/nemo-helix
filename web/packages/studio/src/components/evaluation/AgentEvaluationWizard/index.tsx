// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { isConflictError } from '@nemo/common/src/api/common/utils';
import { parseFilesetLocation } from '@nemo/common/src/components/DatasetFileSelect/parseFilesetLocation';
import { ControlledCheckbox } from '@nemo/common/src/components/form/ControlledCheckbox';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { DEFAULT_DEBOUNCE_MS } from '@nemo/common/src/constants';
import { getURNFromNamedEntityRef } from '@nemo/common/src/namedEntity';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { getEntityNameError, toValidEntityName } from '@nemo/common/src/utils/entityName';
import { InfoCircle } from '@nemo/icons/icons';
import { evalsCreateEvaluateJob } from '@nemo/sdk/generated/evals/evals-plugin-jobs-routes';
import type { AgentEvaluateJobRequest, EvaluateJobRequest } from '@nemo/sdk/generated/evals/schema';
import { deleteEvaluation, useListEvaluations } from '@nemo/sdk/generated/platform/evaluations';
import {
  createExperiment,
  deleteExperiment,
  useListExperiments,
} from '@nemo/sdk/generated/platform/experiments';
import {
  filesCreateFileset,
  filesDeleteFileset,
  filesDownloadFile,
  filesUploadFile,
  useFilesListFilesetFiles,
} from '@nemo/sdk/generated/platform/files';
import {
  Anchor,
  Banner,
  Button,
  Flex,
  FormField,
  RadioGroup,
  SegmentedControl,
  Stack,
  Text,
  Tooltip,
  Upload,
} from '@nvidia/foundations-react-core';
import { submitAgentEvalJob } from '@studio/api/evaluation/agent-evaluations';
import { useFilesetFile } from '@studio/api/files/useFilesetFile';
import {
  buildAgentEvaluationSpec,
  configFitIssues,
  defaultAgentPrompt,
  defaultJudgePrompt,
} from '@studio/components/evaluation/AgentEvaluationWizard/agentEvalConfig';
import {
  DATASET_FILE_TYPES,
  EVAL_CONFIG_FILE_ACCEPT,
} from '@studio/components/evaluation/AgentEvaluationWizard/consts';
import { EvaluationSourceSelect } from '@studio/components/evaluation/AgentEvaluationWizard/EvaluationSourceSelect';
import { FilesetConfigPicker } from '@studio/components/evaluation/AgentEvaluationWizard/FilesetConfigPicker';
import { inspectDatasetFile } from '@studio/components/evaluation/AgentEvaluationWizard/inspectDatasetFile';
import { parquetBatchGroups } from '@studio/components/evaluation/AgentEvaluationWizard/parquetBatchGroups';
import { PromptTemplateSection } from '@studio/components/evaluation/AgentEvaluationWizard/PromptTemplateSection';
import { useEvaluationSources } from '@studio/components/evaluation/AgentEvaluationWizard/useEvaluationSources';
import {
  type EvaluationMode,
  isLastStep,
  nextStep,
  previousStep,
  stepHeading,
  stepIndex,
  stepsFor,
  type WizardStep,
} from '@studio/components/evaluation/AgentEvaluationWizard/wizardSteps';
import { EvalWizard } from '@studio/components/evaluation/EvalWizard';
import { DatasetPanel } from '@studio/components/evaluation/EvalWizard/DatasetPanel';
import {
  entityNameField,
  submitErrorMessage,
  unsalvageableNameError,
} from '@studio/components/evaluation/EvalWizard/entityNameField';
import { hasMalformedBraces } from '@studio/components/evaluation/EvalWizard/judgePromptVariables';
import { EVALUATION_FORM_DEFAULTS } from '@studio/components/evaluation/EvalWizard/types';
import { useDatasetPreview } from '@studio/components/evaluation/EvalWizard/useDatasetPreview';
import {
  createRunEvaluation,
  evalConfigFilename,
  evaluationConfigError,
  evaluationFilesetName,
  findEvalConfigFile,
} from '@studio/components/evaluation/experimentEvalConfig';
import {
  EXPERIMENT_SETTINGS_DEFAULTS,
  experimentSettingsPayload,
  experimentSettingsSchemaShape,
} from '@studio/components/evaluation/experimentSettings';
import { ExperimentSettingsFields } from '@studio/components/evaluation/ExperimentSettingsFields';
import { ScoreDefinitions } from '@studio/components/evaluation/Jobs/form/ScoreDefinitions';
import type { PanelScoreFormData } from '@studio/components/evaluation/Jobs/form/ScoreModal';
import { JudgeModelSelect } from '@studio/components/evaluation/JudgeModelSelect';
import {
  bareName,
  buildAgentEvalRequestBody,
  buildDatasetEvalRequestBody,
  DEFAULT_PARALLELISM,
  type DatasetEvalSpec,
  type EvalConfigFormat,
  type EvalSpec,
  filesetNameForExperiment,
  injectJudgeModel,
  isDatasetEvalSpec,
  MODE_DEFAULT,
  MODE_EXPERIMENT,
  parseEvalConfig,
  parseUploadedDatasetConfig,
  serializeEvalConfig,
} from '@studio/components/evaluation/submitEvaluationJob';
import { nameCheckStatus, nameFieldSlots } from '@studio/components/form/entityNameCheck';
import { LINK_DOCS_STUDIO_EXPERIMENTS, LINK_EVAL_DOCS } from '@studio/constants/links';
import { useJudgeModels } from '@studio/hooks/evaluation/useJudgeModels';
import { getAgentEvaluationsTabRoute } from '@studio/routes/utils';
import { skipToken, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { type FC, useEffect, useMemo, useRef, useState } from 'react';
import {
  FormProvider,
  type SubmitErrorHandler,
  type SubmitHandler,
  useForm,
  useWatch,
} from 'react-hook-form';
import { useNavigate } from 'react-router';
import { useDebounce } from 'use-debounce';
import { z } from 'zod';

const startItems = (rerunDisabled: boolean) => [
  {
    value: MODE_EXPERIMENT,
    disabled: rerunDisabled,
    children: (
      <Stack gap="density-xs">
        <Flex align="center" gap="density-xs">
          <Text kind="label/bold/md">Re-run an existing evaluation</Text>
          {rerunDisabled && (
            <Tooltip slotContent="No existing evaluations to re-run.">
              <InfoCircle size="14" aria-label="Why is this option disabled?" />
            </Tooltip>
          )}
        </Flex>
        <Text className="text-secondary" kind="body/regular/sm">
          Reuses the eval config saved on a previous run. The new run joins that run&apos;s
          experiment, so the two sit side by side on its leaderboard.
        </Text>
      </Stack>
    ),
  },
  {
    value: MODE_DEFAULT,
    children: (
      <Stack gap="density-xs">
        <Text kind="label/bold/md">Create a new experiment</Text>
        <Text className="text-secondary" kind="body/regular/sm">
          Sets up a fresh experiment to group runs under, then takes a dataset and an eval config to
          measure them with.
        </Text>
      </Stack>
    ),
  },
];

const CONFIG_KIND_FORM = 'form';
const CONFIG_KIND_FILESET = 'fileset';
const CONFIG_KIND_UPLOAD = 'upload';

const CONFIG_KINDS = [CONFIG_KIND_FORM, CONFIG_KIND_FILESET, CONFIG_KIND_UPLOAD] as const;

type ConfigKind = (typeof CONFIG_KINDS)[number];

const CONFIG_KIND_ITEMS = [
  { value: CONFIG_KIND_FORM, children: 'Build with Form' },
  { value: CONFIG_KIND_FILESET, children: 'Choose from Fileset' },
  { value: CONFIG_KIND_UPLOAD, children: 'Upload' },
];

const MAX_PARALLELISM = 16;

const NO_EVALUATIONS_MESSAGE =
  'No evaluations with a reusable eval config yet. Go back and create an experiment instead — its run is re-runnable from here afterwards.';

/** Why the step's primary action is disabled. A reason already shown under its field is not
 *  repeated beside the action. */
type StepBlocker = { kind: 'field' } | { kind: 'hint'; hint: string };

const SHOWN_ON_FIELD: StepBlocker = { kind: 'field' };

const blockedWith = (hint: string): StepBlocker => ({ kind: 'hint', hint });

const submitEvaluationBaseSchema = z.object({
  agent: z.string().min(1, 'Agent is required'),
  judgeModel: z.string(),
  mode: z.enum([MODE_DEFAULT, MODE_EXPERIMENT]),
  /** Name of the experiment created on the "new experiment" path. The fileset holding this run's
   *  eval config and dataset is derived from it. */
  newName: entityNameField(),
  /** Name of the Intake Evaluation this run publishes under. Asked for on both paths: the run's
   *  name is how it is told apart from its siblings on the leaderboard, so it is the natural
   *  place to record what changed ("…-temp-1" vs "…-temp-point5"). */
  evaluationRecordName: entityNameField(),
  /** Name of the existing evaluation whose eval config is reused on the re-run path. */
  evaluationName: z.string(),
  /** Read every Parquet file beside the picked one as one dataset. */
  useAllShards: z.boolean(),
  /** Whether the eval config is built here, picked from a fileset, or uploaded. */
  configKind: z.enum(CONFIG_KINDS),
  /** The prompt sent to the agent; ``null`` while it is still the generated one. */
  promptTemplate: z.string().nullable(),
  dataset: z.string().nullable(),
  /** The guided form's LLM judge. */
  body: z.object({
    judgeModel: z.string(),
    judgePrompt: z.string().nullable(),
    scores: z.array(z.custom<PanelScoreFormData>()),
  }),
  configFileset: z.string(),
  configFile: z.string(),
  parallelism: z.coerce
    .number()
    .int('Use a whole number')
    .min(1, 'At least 1')
    .max(MAX_PARALLELISM, `At most ${MAX_PARALLELISM}`),
  ...experimentSettingsSchemaShape,
});

type SubmitEvaluationFormData = z.infer<typeof submitEvaluationBaseSchema>;

const isUnusablePrompt = (prompt: string | null) =>
  prompt !== null && (!prompt.trim() || hasMalformedBraces(prompt));

const makeSubmitEvaluationSchema = (requiresJudgeModel: () => boolean) =>
  submitEvaluationBaseSchema.superRefine((data, ctx) => {
    if (requiresJudgeModel() && !data.judgeModel) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'Select a model to override the config',
        path: ['judgeModel'],
      });
    }
    if (data.mode === MODE_DEFAULT) {
      // Values here are already sanitized by the field transform, so the only naming failures
      // left are "nothing salvageable" and the derived fileset name — which is longer, so a name
      // that is legal on its own can still overflow once "-data" is appended, and there is no
      // fileset field to correct it in.
      const nameError =
        unsalvageableNameError(data.newName, 'Experiment name') ??
        getEntityNameError(filesetNameForExperiment(data.newName), 'Derived fileset name');
      if (nameError) {
        ctx.addIssue({ code: z.ZodIssueCode.custom, message: nameError, path: ['newName'] });
      }
      if (!data.dataset) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'Select an input file.',
          path: ['dataset'],
        });
      }
      if (data.configKind === CONFIG_KIND_FORM) {
        if (isUnusablePrompt(data.promptTemplate)) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: 'Write a prompt.',
            path: ['promptTemplate'],
          });
        }
        if (isUnusablePrompt(data.body.judgePrompt)) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: 'Write a prompt.',
            path: ['body', 'judgePrompt'],
          });
        }
        if (!data.body.judgeModel) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: 'Select a judge model.',
            path: ['body', 'judgeModel'],
          });
        }
        if (data.body.scores.length === 0) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: 'Add at least one score for the judge to extract.',
            path: ['body', 'scores'],
          });
        }
      }
    }
    if (data.mode === MODE_EXPERIMENT && !data.evaluationName) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'Pick an evaluation to re-run',
        path: ['evaluationName'],
      });
    }
    const recordNameError = unsalvageableNameError(data.evaluationRecordName, 'Evaluation name');
    if (recordNameError) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: recordNameError,
        path: ['evaluationRecordName'],
      });
    }
  });

const makeDefaultValues = (
  agent?: string,
  sourceEvaluation?: string
): SubmitEvaluationFormData => ({
  agent: agent ?? '',
  judgeModel: '',
  // Provisional: the preselect effect corrects this once the evaluation list resolves.
  mode: MODE_EXPERIMENT,
  newName: '',
  evaluationRecordName: '',
  evaluationName: sourceEvaluation ?? '',
  useAllShards: false,
  configKind: CONFIG_KIND_FORM,
  promptTemplate: null,
  dataset: null,
  body: { judgeModel: '', judgePrompt: null, scores: EVALUATION_FORM_DEFAULTS.body.scores },
  configFileset: '',
  configFile: '',
  parallelism: DEFAULT_PARALLELISM,
  ...EXPERIMENT_SETTINGS_DEFAULTS,
});

/** A file the user picked, plus why it was rejected when it was. */
interface FilePick {
  file: File;
  error?: string;
}

/** The picked eval config, with its parsed spec once it validates. ``format`` follows the
 *  uploaded extension, not the detected syntax: a file named .yaml is stored as YAML even if
 *  its contents happen to be valid JSON, which is the mapping an author expects. */
interface ConfigPick extends FilePick {
  spec?: DatasetEvalSpec;
  format?: EvalConfigFormat;
}

const configFormatForFile = (name: string): EvalConfigFormat =>
  /\.ya?ml$/i.test(name) ? 'yaml' : 'json';

const isConcreteModel = (model: unknown): model is { url: string; name: string } =>
  typeof model === 'object' &&
  model !== null &&
  'url' in model &&
  typeof model.url === 'string' &&
  'name' in model &&
  typeof model.name === 'string';

const readConfigPick = async (file: File): Promise<ConfigPick> => {
  try {
    return {
      file,
      spec: parseUploadedDatasetConfig(await file.text()),
      format: configFormatForFile(file.name),
    };
  } catch (err) {
    return { file, error: err instanceof Error ? err.message : 'Could not read the file' };
  }
};

interface AgentEvaluationWizardProps {
  workspace: string;
  /** The agent under evaluation. */
  agent: string;
  /** Name of an evaluation to start from. Opens the wizard on its last step with that run already
   *  chosen, which is how "New evaluation from this configuration" hands a row's config over. */
  sourceEvaluation?: string;
}

interface SeededEntities {
  filesetName?: string;
  experimentName?: string;
  evaluationName?: string;
}

/** Undo what a failed submit created, and return the error to raise.
 *
 *  Deleting the Experiment soft-deletes the Evaluations whose only membership was that group, so
 *  the evaluation is only deleted on its own when the experiment pre-existed this submit and is
 *  therefore being kept. Names are freed either way, since the API renames on delete. When a
 *  delete itself fails the returned error names what to remove by hand. */
const discardSeeded = async (
  workspace: string,
  seeded: SeededEntities,
  cause: unknown
): Promise<unknown> => {
  const leftovers: string[] = [];
  const signal = new AbortController().signal;
  if (seeded.experimentName) {
    await deleteExperiment(workspace, seeded.experimentName, signal).catch(() =>
      leftovers.push(`experiment "${seeded.experimentName}"`)
    );
  } else if (seeded.evaluationName) {
    await deleteEvaluation(workspace, seeded.evaluationName, signal).catch(() =>
      leftovers.push(`evaluation "${seeded.evaluationName}"`)
    );
  }
  if (seeded.filesetName) {
    await filesDeleteFileset(workspace, seeded.filesetName, signal).catch(() =>
      leftovers.push(`fileset "${seeded.filesetName}"`)
    );
  }
  if (leftovers.length === 0) return cause;
  const causeDetail = cause instanceof Error ? cause.message : String(cause);
  return new Error(
    `${causeDetail} — ${leftovers.join(' and ')} could not be removed; delete manually before retrying under the same name.`,
    { cause }
  );
};

/** The eval config of the "new experiment" path, built or picked, and the dataset it reads. */
interface EvalInputs {
  spec: DatasetEvalSpec;
  /** Serialization the config was uploaded in; the stored file keeps it. */
  configFormat: EvalConfigFormat;
  /** The dataset, read in place: a file, or a glob over Parquet shards. */
  datasetRef: string;
}

/** Resolves the persisted yardstick spec for this submission. On the "new experiment" path it
 *  takes the built or picked config, bakes in the picked judge and the dataset ref, and seeds it
 *  into a new fileset; on the re-run path it reads the saved spec back verbatim (no re-bake, no
 *  judge re-pick). */
const loadPersistedSpec = async (
  workspace: string,
  formData: SubmitEvaluationFormData,
  configFileset: string | null,
  inputs: EvalInputs | null
): Promise<EvalSpec> => {
  if (formData.mode === MODE_DEFAULT) {
    if (!inputs) throw new Error('Pick a dataset and an eval config before submitting');
    const signal = new AbortController().signal;
    const name = filesetNameForExperiment(formData.newName);
    const judgeModel = formData.judgeModel || null;
    const spec: DatasetEvalSpec = {
      ...inputs.spec,
      dataset: inputs.datasetRef,
      metrics: judgeModel
        ? inputs.spec.metrics.map((m) => injectJudgeModel(m, judgeModel))
        : inputs.spec.metrics,
    };

    try {
      await filesCreateFileset(workspace, { name, description: 'Agent Evaluation Config' }, signal);
    } catch (err) {
      if (isConflictError(err)) {
        throw new Error(
          `A fileset named "${name}" already exists — choose a different experiment name`
        );
      }
      throw err;
    }
    try {
      await filesUploadFile(
        workspace,
        name,
        evalConfigFilename(inputs.configFormat),
        new Blob([serializeEvalConfig(spec, inputs.configFormat)], {
          type: inputs.configFormat === 'yaml' ? 'application/yaml' : 'application/json',
        }),
        signal
      );
    } catch (uploadErr) {
      throw await discardSeeded(workspace, { filesetName: name }, uploadErr);
    }
    return spec;
  }
  if (!configFileset) throw new Error('The selected evaluation has no eval config fileset');
  const signal = new AbortController().signal;
  // The stored config keeps whichever serialization its author uploaded, so resolve the file
  // by listing rather than assuming an extension. parseEvalConfig reads either.
  const configFile = await findEvalConfigFile(workspace, configFileset, signal);
  if (!configFile) throw new Error("Failed to read the selected evaluation's eval config");
  const blob = await filesDownloadFile(workspace, configFileset, configFile, signal);
  if (!blob) throw new Error("Failed to read the selected evaluation's eval config");
  return parseEvalConfig(await blob.text());
};

export const AgentEvaluationWizard: FC<AgentEvaluationWizardProps> = ({
  workspace,
  agent: agentProp,
  sourceEvaluation,
}) => {
  const toast = useToast();
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  // Ref keeps the override's required-ness current for the zod schema getter at validation time.
  const judgeRequiredRef = useRef(false);
  // Latches once the evaluation list resolves; after that the field is the user's.
  const modeDefaultApplied = useRef(false);
  const [schema] = useState(() => makeSubmitEvaluationSchema(() => judgeRequiredRef.current));

  // A handed-in source has already answered the first two steps, so the wizard opens on the last
  // one. Back still walks all the way out, so the choice stays reviewable rather than assumed.
  const [step, setStep] = useState<WizardStep>(sourceEvaluation ? 'evaluation' : 'start');
  const [submitAttempted, setSubmitAttempted] = useState(false);
  const [configPick, setConfigPick] = useState<ConfigPick | null>(null);

  // Bumped whenever a pick is replaced, removed, or reset, so an async validation that is
  // still running when that happens knows to drop its result instead of committing it.
  const configToken = useRef(0);

  const methods = useForm<SubmitEvaluationFormData>({
    resolver: zodResolver(schema),
    defaultValues: makeDefaultValues(agentProp, sourceEvaluation),
    mode: 'onSubmit',
    reValidateMode: 'onChange',
  });
  const { control, setValue, handleSubmit, clearErrors, trigger, formState } = methods;
  const { errors } = formState;

  const mode = useWatch({ control, name: 'mode' }) as EvaluationMode;
  const selectedAgent = useWatch({ control, name: 'agent' });
  const steps = stepsFor(mode);

  const evaluationName = useWatch({ control, name: 'evaluationName' });

  // Every reusable evaluation for this agent, resolved to its experiment. One list feeds the
  // picker, its section headings, and the lookup that resolves the chosen run.
  const sources = useEvaluationSources({
    workspace,
    agent: selectedAgent || undefined,
    enabled: !!selectedAgent,
  });
  const selectedSource = sources.byName[evaluationName];
  const selectedEvaluation = selectedSource?.evaluation;
  const hasNoEvaluations = mode === MODE_EXPERIMENT && sources.isEmpty;
  // Only a settled, agent-scoped, genuinely empty list disables re-run: a disabled query reports
  // isLoading false, which would read as empty before anything was fetched.
  const rerunUnavailable = !!selectedAgent && !sources.isLoading && sources.isEmpty;

  // Default the start mode once the list resolves; `selectedAgent` gates it because a disabled
  // query reports isLoading false, which would read as empty before anything was fetched.
  useEffect(() => {
    if (!selectedAgent || sources.isLoading || modeDefaultApplied.current) return;
    modeDefaultApplied.current = true;
    if (sources.isEmpty) setValue('mode', MODE_DEFAULT);
  }, [selectedAgent, sources.isLoading, sources.isEmpty, setValue]);

  // Nothing is preselected here. A step whose whole job is "choose the run to re-run" should not
  // answer itself — Next stays disabled until the user picks, and stepBlocker says why. (A run
  // handed in by "New evaluation from this configuration" arrives in the form's default values,
  // which is a choice already made rather than one guessed at.)

  const { data: evaluationConfigIssue, isFetching: isValidatingEvaluation } = useQuery({
    queryKey: ['evaluation-eval-config', workspace, evaluationName],
    queryFn: ({ signal }) =>
      selectedEvaluation ? evaluationConfigError(workspace, selectedEvaluation, signal) : null,
    enabled: mode === MODE_EXPERIMENT && !!selectedEvaluation,
  });

  const evaluationFileset = selectedEvaluation ? evaluationFilesetName(selectedEvaluation) : null;
  const evaluationFieldError = errors.evaluationName?.message ?? evaluationConfigIssue ?? undefined;

  const configKind = useWatch({ control, name: 'configKind' });
  const isFormConfig = configKind === CONFIG_KIND_FORM;
  const [dataset, promptTemplate, useAllShards, judgePrompt] = useWatch({
    control,
    name: ['dataset', 'promptTemplate', 'useAllShards', 'body.judgePrompt'],
  });
  const datasetLocation = dataset ? parseFilesetLocation(dataset) : null;
  // Reading the sibling shards is opted into per file: another pick may share a folder with
  // files that are not part of it, such as a test split beside a train split.
  useEffect(() => setValue('useAllShards', false), [dataset, setValue]);
  const preview = useDatasetPreview(dataset ?? null);
  const columns = useMemo(
    () => preview.keyOptions.map((option) => option.value),
    [preview.keyOptions]
  );
  // The preview reads only what it needs to show a row, so the whole file is read here to hold it
  // to the evaluator's loader before the run is created.
  const datasetFile = useFilesetFile({
    workspace: datasetLocation?.workspace ?? workspace,
    fileset: datasetLocation?.name ?? '',
    path: datasetLocation?.objectPath ?? '',
    enabled: !!datasetLocation,
  });
  const downloadedDataset = datasetFile.data;
  const datasetInspection = useQuery({
    queryKey: ['evaluation-dataset-pick', dataset, datasetFile.dataUpdatedAt],
    queryFn: downloadedDataset ? () => inspectDatasetFile(downloadedDataset) : skipToken,
    gcTime: 0,
  });
  const isReadingDataset =
    preview.isLoading || datasetFile.isFetching || datasetInspection.isFetching;
  const datasetError =
    datasetInspection.data?.error ?? datasetFile.error?.message ?? preview.error ?? undefined;

  // Parquet shards beside the picked file, read together through one glob when the user keeps
  // the box ticked.
  const filesetFiles = useFilesListFilesetFiles(
    datasetLocation?.workspace ?? '',
    datasetLocation?.name ?? '',
    undefined,
    { query: { enabled: !!datasetLocation } }
  );
  const pickedDir = datasetLocation?.objectPath.slice(
    0,
    datasetLocation.objectPath.lastIndexOf('/') + 1
  );
  const shards = datasetLocation?.objectPath.endsWith('.parquet')
    ? parquetBatchGroups((filesetFiles.data?.data ?? []).map((file) => file.path)).find(
        (group) => group.dir === pickedDir
      )
    : undefined;
  const datasetRef =
    datasetLocation && shards && useAllShards
      ? `${datasetLocation.workspace}/${datasetLocation.name}#${shards.glob}`
      : dataset;

  const configFileset = useWatch({ control, name: 'configFileset' });
  const configFilePath = useWatch({ control, name: 'configFile' });
  const isConfigFromFileset = configKind === CONFIG_KIND_FILESET;
  const configFilesetFile = useFilesetFile({
    workspace,
    fileset: configFileset,
    path: configFilePath,
    enabled: isConfigFromFileset,
  });
  const downloadedConfig = configFilesetFile.data;
  const filesetConfigRead = useQuery({
    queryKey: [
      'evaluation-config-pick',
      workspace,
      configFileset,
      configFilePath,
      configFilesetFile.dataUpdatedAt,
    ],
    queryFn: downloadedConfig ? () => readConfigPick(downloadedConfig) : skipToken,
    gcTime: 0,
  });
  const isLoadingFilesetConfig = configFilesetFile.isFetching || filesetConfigRead.isFetching;
  const activeConfigPick = isConfigFromFileset ? (filesetConfigRead.data ?? null) : configPick;

  const pickedSpec = activeConfigPick?.error ? undefined : activeConfigPick?.spec;

  // Checked against the row the preview read, so nothing is reported before the columns are known.
  const configIssues =
    !isFormConfig && pickedSpec && preview.row ? configFitIssues(pickedSpec, columns) : [];
  const generatedAgentPrompt = defaultAgentPrompt(columns);
  const generatedJudgePrompt = defaultJudgePrompt(columns);
  const promptFitIssues = (prompt: string) =>
    isFormConfig && preview.row
      ? configFitIssues({ dataset: '', metrics: [], prompt_template: prompt }, columns)
      : [];
  const promptIssues = promptFitIssues(promptTemplate ?? generatedAgentPrompt);
  const judgePromptIssues = promptFitIssues(judgePrompt ?? generatedJudgePrompt);

  const rawExperimentName = useWatch({ control, name: 'newName' });
  const rawRecordName = useWatch({ control, name: 'evaluationRecordName' });
  const experimentPreview = toValidEntityName(rawExperimentName, '');
  const recordPreview = toValidEntityName(rawRecordName, '');

  const [debouncedExperimentName] = useDebounce(experimentPreview, DEFAULT_DEBOUNCE_MS);
  const [debouncedRecordName] = useDebounce(recordPreview, DEFAULT_DEBOUNCE_MS);
  const isCreateMode = mode === MODE_DEFAULT;

  // Prefill the run's name with the source's own; seeded once per source, never over a typed name.
  const suggestedRecordName = useRef('');
  const seededForSource = useRef<string | null>(null);
  useEffect(() => {
    if (mode !== MODE_EXPERIMENT || !evaluationName) return;
    if (seededForSource.current === evaluationName) return;
    seededForSource.current = evaluationName;
    if (rawRecordName && rawRecordName !== suggestedRecordName.current) return;
    suggestedRecordName.current = evaluationName;
    setValue('evaluationRecordName', evaluationName, { shouldValidate: false });
  }, [mode, evaluationName, rawRecordName, setValue]);

  // Leaving the re-run path retires the derived name. It is borrowed from whichever run was
  // selected there, and carrying it onto a brand-new experiment's first evaluation would name that
  // run after an unrelated one. Only the suggestion is dropped — a name the user typed is theirs,
  // on either path — and clearing the seed marker lets a switch back suggest again.
  useEffect(() => {
    if (mode !== MODE_DEFAULT) return;
    if (!rawRecordName || rawRecordName !== suggestedRecordName.current) return;
    suggestedRecordName.current = '';
    seededForSource.current = null;
    modeDefaultApplied.current = false;
    setValue('evaluationRecordName', '');
  }, [mode, rawRecordName, setValue]);

  const experimentConflictQuery = useListExperiments(
    workspace,
    { page_size: 1, filter: { name: debouncedExperimentName } },
    { query: { enabled: isCreateMode && !!debouncedExperimentName } }
  );
  const recordConflictQuery = useListEvaluations(
    workspace,
    { page_size: 1, filter: { name: debouncedRecordName } },
    { query: { enabled: !!debouncedRecordName } }
  );

  const configError =
    activeConfigPick?.error ??
    (isConfigFromFileset ? configFilesetFile.error?.message : undefined) ??
    (submitAttempted && !activeConfigPick && !isLoadingFilesetConfig
      ? 'Select an eval configuration'
      : undefined);

  const experimentNameStatus = nameCheckStatus(
    experimentPreview,
    debouncedExperimentName,
    experimentConflictQuery
  );
  const recordNameStatus = nameCheckStatus(recordPreview, debouncedRecordName, recordConflictQuery);

  const experimentNameSlots = nameFieldSlots({
    entity: 'experiment',
    preview: experimentPreview,
    status: experimentNameStatus,
    schemaError: errors.newName?.message,
    describe: 'Groups multiple evaluation runs together for comparison.',
  });

  const recordNameSlots = nameFieldSlots({
    entity: 'evaluation',
    preview: recordPreview,
    status: recordNameStatus,
    schemaError: errors.evaluationRecordName?.message,
    describe: isCreateMode
      ? 'Names this run within the experiment. Results publish under it.'
      : 'Names this run alongside the one it re-runs — say what changed, e.g. a new temperature.',
  });

  // Mirrors injectJudgeModel's own guard, so the picker is shown exactly when a metric would
  // have a model written into it — not just for the llm-judge type. Every match is collected,
  // not just the first: the override rewrites all of them, so all of them must be checked.
  const judgeMetrics =
    activeConfigPick?.spec?.metrics.filter(
      (metric) => metric.metric_type === 'llm-judge' || 'model' in metric.payload.metric
    ) ?? [];

  const isLlmJudge = mode === MODE_DEFAULT && !isFormConfig && judgeMetrics.length > 0;

  // A different query from the one behind the dropdown (lazy, paginated for search); this is the
  // only complete workspace list, so it decides what counts as valid.
  const { data: judgeModels, isLoading: isJudgeModelsLoading } = useJudgeModels({});

  const workspaceModelUrns = new Set<string>(
    judgeModels
      ?.map((model) => getURNFromNamedEntityRef(model))
      .filter((urn): urn is NonNullable<typeof urn> => urn !== undefined) ?? []
  );

  // A ref string is checked against the workspace below, and a concrete `{url, name}` Model (what
  // the model wizard writes) runs as-is. Anything else cannot run, so it requires the override.
  const hasModellessJudge =
    isLlmJudge &&
    judgeMetrics.some((metric) => {
      const model = metric.payload.metric.model;
      return typeof model !== 'string' && !isConcreteModel(model);
    });

  // Full `workspace/name` ModelRefs: an unqualified bare name is unreachable to the evaluator's
  // resolver. Empty while loading, or every model would read as unusable.
  const invalidModelRefs =
    isLlmJudge && !isJudgeModelsLoading
      ? [
          ...new Set(
            judgeMetrics
              .map((metric) => metric.payload.metric.model)
              .filter((model): model is string => typeof model === 'string')
              .filter((model) => !workspaceModelUrns.has(model))
          ),
        ]
      : [];

  const judgeModel = useWatch({ control, name: 'judgeModel' });
  const invalidModelsError =
    invalidModelRefs.length > 0 && !judgeModel
      ? `The following llm judge models in the config are not valid in this workspace: [${invalidModelRefs.join(', ')}]`
      : undefined;

  const judgeRequired = isLlmJudge && (invalidModelRefs.length > 0 || hasModellessJudge);
  judgeRequiredRef.current = judgeRequired;

  const clearConfigPick = () => {
    configToken.current += 1;
    setConfigPick(null);
  };

  // Clears first: validation is async, and leaving the previous pick in place would keep the
  // submit gate open against a file the user has already replaced. It also drops its result if
  // the pick is no longer current — a slow read of a replaced file would otherwise land after a
  // faster one and submit a file the card no longer shows.
  // A fresh config also means a fresh judge: the preselect effect above bails once judgeModel
  // is set, so leaving it would keep the previous file's judge.
  const handleConfigPicked = async (item?: { file: File }) => {
    setValue('judgeModel', '');
    clearConfigPick();
    if (!item?.file) return;
    const token = configToken.current;
    const pick = await readConfigPick(item.file);
    if (token !== configToken.current) return;
    setConfigPick(pick);
  };

  // The override belongs to a picked config; a built one picks its own judge.
  const handleConfigKindChanged = (value: string) => {
    setValue('judgeModel', '');
    clearErrors();
    clearConfigPick();
    setValue('configFileset', '');
    setValue('configFile', '');
    setValue('configKind', value as ConfigKind);
  };

  const {
    mutateAsync: submitEvaluation,
    error: submitError,
    isPending,
  } = useMutation({
    mutationFn: async (formData: SubmitEvaluationFormData) => {
      const configInputs =
        formData.configKind === CONFIG_KIND_FORM
          ? {
              spec: buildAgentEvaluationSpec(
                formData.dataset ?? '',
                formData.promptTemplate ?? generatedAgentPrompt,
                {
                  ...formData.body,
                  judgePrompt: formData.body.judgePrompt ?? generatedJudgePrompt,
                },
                workspace
              ),
              configFormat: 'json' as const,
            }
          : pickedSpec && { spec: pickedSpec, configFormat: activeConfigPick?.format ?? 'json' };
      const inputs = configInputs && datasetRef ? { ...configInputs, datasetRef } : null;
      const spec = await loadPersistedSpec(workspace, formData, evaluationFileset, inputs);

      const isNew = formData.mode === MODE_DEFAULT;
      const filesetName = isNew
        ? filesetNameForExperiment(formData.newName)
        : (evaluationFileset ?? '');

      const seeded: SeededEntities = isNew ? { filesetName } : {};

      try {
        // The "new experiment" path creates the ExperimentGroup this run lands in, carrying every
        // setting the Experiments page would have offered; the re-run path reuses the picked
        // evaluation's group(s) and records the lineage.
        let experimentIds: string[];
        let nameStem: string;
        let parentEvaluationId: string | undefined;
        if (isNew) {
          const experiment = await createExperiment(workspace, {
            name: formData.newName,
            ...experimentSettingsPayload(formData),
          });
          seeded.experimentName = experiment.name;
          experimentIds = [experiment.id];
          nameStem = experiment.name;
        } else {
          if (!selectedEvaluation) throw new Error('No evaluation to re-run');
          experimentIds = selectedEvaluation.experiment_ids;
          nameStem = selectedSource?.experimentName ?? selectedEvaluation.name;
          parentEvaluationId = selectedEvaluation.id;
        }

        const evaluationId = await createRunEvaluation(workspace, {
          experimentIds,
          name: formData.evaluationRecordName,
          nameStem,
          filesetName,
          datasetName: isNew ? datasetLocation?.name : undefined,
          parentEvaluationId,
        }).catch((err: unknown) => {
          if (isConflictError(err)) {
            throw new Error(
              `An evaluation named "${formData.evaluationRecordName}" already exists — choose a different evaluation name`
            );
          }
          throw err;
        });
        seeded.evaluationName = evaluationId;

        const selections = {
          workspace,
          agent: formData.agent,
          filesetName,
          experimentName: nameStem,
          evaluationId,
          parallelism: formData.parallelism,
        };
        const created = isDatasetEvalSpec(spec)
          ? await evalsCreateEvaluateJob(
              workspace,
              buildDatasetEvalRequestBody(spec, selections, null) as EvaluateJobRequest
            )
          : await submitAgentEvalJob(
              workspace,
              buildAgentEvalRequestBody(spec, selections) as AgentEvaluateJobRequest
            );
        if (!created?.name) throw new Error('Submission did not return a job name');
        return { name: created.name };
      } catch (err) {
        throw await discardSeeded(workspace, seeded, err);
      }
    },
    onSuccess: ({ name }, formData) => {
      toast.success(`Evaluation "${name}" submitted`);
      void queryClient.invalidateQueries({ queryKey: ['evaluator-jobs', workspace] });
      navigate(getAgentEvaluationsTabRoute(workspace, bareName(formData.agent)), {
        replace: true,
      });
    },
  });

  // What the current step still needs before Next means anything. Kept separate from the zod
  // schema: the schema judges the whole submission, and a step must only answer for its own
  // fields — otherwise step one would refuse to advance over a field two steps away.
  const stepBlocker = (): StepBlocker | undefined => {
    if (step === 'start') {
      return selectedAgent ? undefined : blockedWith('Pick an agent to evaluate.');
    }
    if (step === 'experiment') {
      if (!experimentPreview) {
        return blockedWith('Name the experiment to continue.');
      }
      if (experimentNameStatus === 'checking' || experimentNameStatus === 'conflict') {
        return SHOWN_ON_FIELD;
      }
      return errors.newName ? SHOWN_ON_FIELD : undefined;
    }
    if (step === 'configuration') {
      if (isReadingDataset || isLoadingFilesetConfig) {
        return blockedWith('Reading the selected file...');
      }
      if (datasetError) {
        return blockedWith(datasetError);
      }
      if (configIssues.length > 0) {
        return SHOWN_ON_FIELD;
      }
      if (isLlmJudge && isJudgeModelsLoading) {
        return blockedWith('Loading judge models...');
      }
      if (promptIssues.length > 0 || judgePromptIssues.length > 0) {
        return SHOWN_ON_FIELD;
      }
    }
    // Everything Run Evaluation waits on, so the button is disabled instead of swallowing the
    // click. Schema errors stay out: those only show once a submit is tried.
    if (step === 'evaluation') {
      if (mode === MODE_EXPERIMENT) {
        if (hasNoEvaluations) {
          return SHOWN_ON_FIELD;
        }
        // Nothing can be picked, and a handed-in source cannot be resolved or checked, until the
        // list arrives.
        if (sources.isLoading) {
          return blockedWith('Loading evaluations...');
        }
        if (evaluationName && !selectedEvaluation) {
          return blockedWith(`Evaluation ${evaluationName} no longer exists. Pick another.`);
        }
        if (isValidatingEvaluation) {
          return blockedWith('Checking the saved eval config...');
        }
        if (evaluationConfigIssue) {
          return SHOWN_ON_FIELD;
        }
      }
      // invalidModelRefs is empty while the list is in flight, so submitting inside that window
      // would skip the judge-model check entirely.
      if (mode === MODE_DEFAULT && isLlmJudge && isJudgeModelsLoading) {
        return blockedWith('Loading judge models...');
      }
      if (recordNameStatus === 'checking' || recordNameStatus === 'conflict') {
        return SHOWN_ON_FIELD;
      }
      // The experiment's name field is a step back, so its conflict is spelled out here.
      if (isCreateMode && experimentNameStatus === 'conflict') {
        return blockedWith(`An experiment named ${experimentPreview} already exists.`);
      }
    }
    return undefined;
  };
  const blocker = stepBlocker();

  const goNext = async () => {
    if (blocker) {
      return;
    }
    if (step === 'configuration') {
      setSubmitAttempted(true);
      const valid = await trigger(
        isFormConfig
          ? ['parallelism', 'dataset', 'promptTemplate', 'body']
          : ['parallelism', 'dataset', 'judgeModel']
      );
      if (!valid || (!isFormConfig && !pickedSpec)) {
        return;
      }
    }
    setStep(nextStep(steps, step));
  };

  const goBack = () => {
    clearErrors();
    setStep(previousStep(steps, step));
  };

  const onSubmit: SubmitHandler<SubmitEvaluationFormData> = async (formData) => {
    // EvalWizard already drops clicks and Enter while blocked; this keeps a submit from ever
    // skipping a gate.
    if (blocker) {
      return;
    }
    if (
      mode === MODE_DEFAULT &&
      ((!isFormConfig && !pickedSpec) ||
        configIssues.length > 0 ||
        promptIssues.length > 0 ||
        judgePromptIssues.length > 0)
    ) {
      setStep('configuration');
      return;
    }
    try {
      await submitEvaluation(formData);
    } catch {
      // Error rendered via errorText prop.
    }
  };

  const onInvalid: SubmitErrorHandler<SubmitEvaluationFormData> = (invalid) => {
    const failed = (fields: string[]) => fields.some((field) => field in invalid);
    if (
      steps.includes('configuration') &&
      failed(['parallelism', 'dataset', 'promptTemplate', 'body', 'judgeModel'])
    ) {
      setStep('configuration');
    }
  };

  const onLastStep = isLastStep(steps, step);

  // Enter on an intermediate step means "next", not "submit the half-filled form".
  const handleContinue = () => {
    if (!onLastStep) {
      void goNext();
      return;
    }
    setSubmitAttempted(true);
    void handleSubmit(onSubmit, onInvalid)();
  };

  const errorMessage = submitErrorMessage(submitError);

  const parallelismField = (
    <ControlledTextInput
      useControllerProps={{ control, name: 'parallelism' }}
      name="parallelism"
      label={<Text kind="label/bold/xl">Parallel Requests</Text>}
      aria-label="Parallel Requests"
      type="number"
      formFieldProps={{
        slotInfo:
          'Rows sent to the agent at once. Lower it for a slow agent: a deployment that answers one request at a time fails rows that wait too long in its queue.',
        slotError: errors.parallelism?.message,
      }}
    />
  );

  return (
    <EvalWizard
      heading="Run Agent Evaluation"
      headingDescription="Score your agent against a dataset."
      steps={steps.map(stepHeading)}
      activeStep={stepIndex(steps, step)}
      onSubmit={(event) => {
        event.preventDefault();
        handleContinue();
      }}
      slotBanner={
        errorMessage ? (
          <Banner kind="inline" status="error">
            {errorMessage}
          </Banner>
        ) : null
      }
      slotFooterStart={
        stepIndex(steps, step) > 0 ? (
          <Button kind="secondary" onClick={goBack} disabled={isPending}>
            Back
          </Button>
        ) : null
      }
      cancelTo={getAgentEvaluationsTabRoute(workspace, agentProp)}
      continueLabel={onLastStep ? 'Run Evaluation' : 'Next'}
      continueLoading={isPending}
      canContinue={!blocker}
      blockedHint={blocker?.kind === 'hint' ? blocker.hint : undefined}
      onContinue={handleContinue}
    >
      <FormProvider {...methods}>
        <Stack gap="density-xl">
          {step === 'start' && (
            <>
              <Text kind="body/regular/md">
                Run evaluation via NeMo Helix Evals built-in runner. Evaluator also supports Harbor
                and Gym as runners.{' '}
                <Anchor
                  kind="inline"
                  textKind="body/regular/md"
                  href={LINK_EVAL_DOCS}
                  target="_blank"
                  rel="noreferrer"
                >
                  Learn more
                </Anchor>
                .
              </Text>

              <FormField slotLabel="How do you want to start?">
                <RadioGroup
                  kind="tile"
                  orientation="vertical"
                  name="mode"
                  value={mode}
                  onValueChange={(value) => {
                    setValue('mode', value as EvaluationMode, { shouldValidate: false });
                    clearErrors(['evaluationName', 'newName']);
                  }}
                  items={startItems(rerunUnavailable)}
                />
              </FormField>
            </>
          )}

          {step === 'experiment' && (
            <>
              <Text kind="label/bold/xl">Experiment</Text>
              <Text className="text-secondary" kind="body/regular/md">
                Learn more in the{' '}
                <Anchor
                  kind="inline"
                  textKind="body/regular/md"
                  href={LINK_DOCS_STUDIO_EXPERIMENTS}
                  target="_blank"
                  rel="noreferrer"
                >
                  experiments documentation
                </Anchor>
                .
              </Text>
              <ControlledTextInput
                useControllerProps={{ control, name: 'newName' }}
                placeholder="e.g. model-update-tests"
                formFieldProps={{
                  slotLabel: 'Name',
                  ...experimentNameSlots,
                }}
              />
              {/* The same settings the Experiments page offers, so starting from here is not a
                  lesser way to create an experiment. */}
              <ExperimentSettingsFields
                control={control}
                names={{
                  description: 'description',
                  defaultSort: 'defaultSort',
                  isFavorite: 'isFavorite',
                  showEvaluationsOverTime: 'showEvaluationsOverTime',
                }}
                disabled={isPending}
                collapseAdvanced
              />
            </>
          )}

          {step === 'configuration' && (
            <>
              <DatasetPanel
                slotLabel={<Text kind="label/bold/xl">Dataset File</Text>}
                acceptedFileTypes={DATASET_FILE_TYPES}
                allowMultipleUpload
                showMapping={false}
              >
                {shards ? (
                  <ControlledCheckbox
                    useControllerProps={{ control, name: 'useAllShards' }}
                    slotLabel={`Use All ${shards.count} Parquet Files in ${shards.dir || 'the Fileset Root'}`}
                  />
                ) : null}

                {dataset ? (
                  <Stack gap="density-md">
                    <Text kind="label/bold/xl">Eval Configuration</Text>
                    <SegmentedControl
                      aria-label="Eval configuration source"
                      size="tiny"
                      className="w-full"
                      value={configKind}
                      onValueChange={handleConfigKindChanged}
                      items={CONFIG_KIND_ITEMS}
                    />
                  </Stack>
                ) : null}

                {dataset && !isFormConfig ? (
                  <Stack gap="density-md">
                    {isConfigFromFileset ? (
                      <FilesetConfigPicker<SubmitEvaluationFormData>
                        workspace={workspace}
                        control={control}
                        filesetName="configFileset"
                        fileName="configFile"
                        onPick={() => setValue('judgeModel', '')}
                        disabled={isPending}
                        loading={isLoadingFilesetConfig}
                        error={configError}
                      />
                    ) : (
                      <Upload
                        attributes={{
                          UploadInputElement: { 'aria-label': 'Eval configuration file' },
                        }}
                        accept={EVAL_CONFIG_FILE_ACCEPT}
                        onValueChange={handleConfigPicked}
                        onFileRemove={clearConfigPick}
                        status={configError ? 'error' : undefined}
                        renderInput={(slotInput) => (
                          <FormField
                            name="evalConfig"
                            slotHelp="Select a JSON or YAML config."
                            slotError={configError}
                            status={configError ? 'error' : undefined}
                          >
                            {configPick ? null : slotInput}
                          </FormField>
                        )}
                      />
                    )}

                    {configIssues.map((issue) => (
                      <Banner key={issue} kind="inline" status="error">
                        {issue}
                      </Banner>
                    ))}
                  </Stack>
                ) : null}
              </DatasetPanel>

              {dataset && isFormConfig ? (
                <>
                  <PromptTemplateSection
                    name="promptTemplate"
                    title="Agent Prompt"
                    titleKind="label/bold/lg"
                    description="Sent to the agent for every row. It starts with every field: remove the ones the agent shouldn't see, such as expected answers."
                    generated={generatedAgentPrompt}
                    columns={columns}
                    issues={promptIssues}
                  />
                  <Stack gap="density-lg">
                    <Text kind="label/bold/lg">LLM-as-a-Judge</Text>
                    <JudgeModelSelect<SubmitEvaluationFormData>
                      formFieldName="body.judgeModel"
                      slotLabel={<Text kind="label/bold/md">Judge Model</Text>}
                      placeholder="Select a judge model"
                    />
                    <PromptTemplateSection
                      name="body.judgePrompt"
                      title="Judge Prompt"
                      titleKind="label/bold/md"
                      description="Sent to the judge with the agent's response for every row. It starts with every field, so the judge can check the response against the expected values."
                      generated={generatedJudgePrompt}
                      columns={columns}
                      issues={judgePromptIssues}
                      withResponse
                    />
                    <Stack gap="density-sm" className="min-w-0">
                      <Text kind="label/bold/md">Score Definitions</Text>
                      <Text kind="body/regular/md" className="text-secondary">
                        Scores extracted from the judge&apos;s output. These also tell the judge
                        what to grade.
                      </Text>
                      <ScoreDefinitions />
                    </Stack>
                  </Stack>
                </>
              ) : null}

              {dataset && !isFormConfig && pickedSpec ? (
                <Stack gap="density-md">
                  <Text kind="label/bold/lg">Metrics</Text>
                  <Text className="text-secondary" kind="body/regular/md">
                    Scored as the eval config defines them, against agent{' '}
                    <code>{bareName(selectedAgent)}</code>. Any target in the config is replaced by
                    this agent.
                  </Text>
                  <Stack gap="density-sm">
                    {pickedSpec?.metrics.map((metric, index) => {
                      const model = metric.payload.metric.model;
                      const judge =
                        typeof model === 'string'
                          ? model
                          : isConcreteModel(model)
                            ? model.name
                            : '';
                      const description = metric.metadata?.description;
                      return (
                        <Stack key={index} gap="density-xs">
                          <Text kind="label/bold/md">
                            {metric.metric_type}
                            {judge ? ` · judge ${judge}` : ''}
                          </Text>
                          {typeof description === 'string' ? (
                            <Text className="text-secondary" kind="body/regular/sm">
                              {description}
                            </Text>
                          ) : null}
                        </Stack>
                      );
                    })}
                  </Stack>
                  {isLlmJudge && (
                    <JudgeModelSelect<SubmitEvaluationFormData>
                      formFieldName="judgeModel"
                      slotLabel={
                        <Text kind="label/bold/md">
                          {judgeRequired
                            ? 'Override All LLM Models'
                            : 'Override All LLM Models (Optional)'}
                        </Text>
                      }
                      slotError={invalidModelsError}
                    />
                  )}
                </Stack>
              ) : null}

              {parallelismField}
            </>
          )}

          {step === 'evaluation' && (
            <>
              {mode === MODE_EXPERIMENT && <Text kind="label/bold/xl">Configuration</Text>}
              {/* Which run to base this one on, and what to call the result, are one decision:
                  the name is derived from the pick, so the two belong on the same screen. */}
              {mode === MODE_EXPERIMENT &&
                (hasNoEvaluations ? (
                  <Text className="text-secondary" kind="body/regular/md">
                    {NO_EVALUATIONS_MESSAGE}
                  </Text>
                ) : (
                  <EvaluationSourceSelect<SubmitEvaluationFormData>
                    name="evaluationName"
                    options={sources.options}
                    groupLabels={sources.groupLabels}
                    byName={sources.byName}
                    isLoading={sources.isLoading}
                    selectedName={evaluationName}
                    slotError={evaluationFieldError}
                    disabled={isPending}
                  />
                ))}

              <ControlledTextInput
                useControllerProps={{ control, name: 'evaluationRecordName' }}
                placeholder={
                  isCreateMode ? 'e.g. initial-baseline' : 'e.g. nemotron-super-3-temp-1'
                }
                formFieldProps={{
                  slotLabel: isCreateMode ? 'Evaluation Name' : 'New Evaluation Name',
                  ...recordNameSlots,
                }}
              />
              {mode === MODE_EXPERIMENT && parallelismField}
            </>
          )}
        </Stack>
      </FormProvider>
    </EvalWizard>
  );
};
