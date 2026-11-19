// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { isConflictError } from '@nemo/common/src/api/common/utils';
import { parseFilesetLocation } from '@nemo/common/src/components/DatasetFileSelect/parseFilesetLocation';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { getURNFromNamedEntityRef } from '@nemo/common/src/namedEntity';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { getEntityNameError } from '@nemo/common/src/utils/entityName';
import { evalsCreateEvaluateJob } from '@nemo/sdk/generated/evals/evals-plugin-jobs-routes';
import type { EvaluateJobRequest } from '@nemo/sdk/generated/evals/schema';
import { deleteEvaluation } from '@nemo/sdk/generated/platform/evaluations';
import {
  filesCreateFileset,
  filesDeleteFileset,
  filesUploadFile,
  useFilesListFilesetFiles,
} from '@nemo/sdk/generated/platform/files';
import {
  Anchor,
  Banner,
  Button,
  Checkbox,
  FormField,
  SegmentedControl,
  Stack,
  Text,
  Upload,
} from '@nvidia/foundations-react-core';
import { useFilesetFile } from '@studio/api/files/useFilesetFile';
import {
  buildAgentEvaluationSpec,
  configFitIssues,
  defaultAgentPrompt,
  defaultJudgePrompt,
  NO_FIT_ISSUES,
  promptFitIssues,
} from '@studio/components/evaluation/AgentEvaluationWizard/agentEvalConfig';
import {
  DATASET_FILE_TYPES,
  EVAL_CONFIG_FILE_ACCEPT,
} from '@studio/components/evaluation/AgentEvaluationWizard/consts';
import {
  DATASET_SNAPSHOT_KEY,
  takeDatasetSnapshot,
} from '@studio/components/evaluation/AgentEvaluationWizard/datasetSnapshot';
import { FilesetConfigPicker } from '@studio/components/evaluation/AgentEvaluationWizard/FilesetConfigPicker';
import { inspectDatasetFile } from '@studio/components/evaluation/AgentEvaluationWizard/inspectDatasetFile';
import { parquetBatchGroups } from '@studio/components/evaluation/AgentEvaluationWizard/parquetBatchGroups';
import { PromptTemplateSection } from '@studio/components/evaluation/AgentEvaluationWizard/PromptTemplateSection';
import {
  resolveRunTarget,
  RUN_TARGET_DEFAULTS,
  runTargetSchemaShape,
  useRunNameCheck,
} from '@studio/components/evaluation/AgentEvaluationWizard/runTarget';
import { RunTargetFields } from '@studio/components/evaluation/AgentEvaluationWizard/RunTargetFields';
import { EvalWizard } from '@studio/components/evaluation/EvalWizard';
import { DatasetPanel } from '@studio/components/evaluation/EvalWizard/DatasetPanel';
import {
  submitErrorMessage,
  unsalvageableNameError,
} from '@studio/components/evaluation/EvalWizard/entityNameField';
import { hasMalformedBraces } from '@studio/components/evaluation/EvalWizard/judgePromptVariables';
import { EVALUATION_FORM_DEFAULTS } from '@studio/components/evaluation/EvalWizard/types';
import { useDatasetPreview } from '@studio/components/evaluation/EvalWizard/useDatasetPreview';
import {
  createRunEvaluation,
  evalConfigFilename,
  recordRunJob,
  runMetadata,
} from '@studio/components/evaluation/experimentEvalConfig';
import { ScoreDefinitions } from '@studio/components/evaluation/Jobs/form/ScoreDefinitions';
import type { PanelScoreFormData } from '@studio/components/evaluation/Jobs/form/ScoreModal';
import { JudgeModelSelect } from '@studio/components/evaluation/JudgeModelSelect';
import {
  bareName,
  buildDatasetEvalRequestBody,
  dataFilesetName,
  DEFAULT_PARALLELISM,
  type DatasetEvalSpec,
  type EvalConfigFormat,
  injectJudgeModel,
  parseUploadedDatasetConfig,
  serializeEvalConfig,
} from '@studio/components/evaluation/submitEvaluationJob';
import { LINK_EVAL_DOCS } from '@studio/constants/links';
import { useJudgeModels } from '@studio/hooks/evaluation/useJudgeModels';
import { getAgentEvaluationsTabRoute } from '@studio/routes/utils';
import { skipToken, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { type FC, useMemo, useRef, useState } from 'react';
import { FormProvider, type SubmitHandler, useForm, useWatch } from 'react-hook-form';
import { useNavigate } from 'react-router';
import { z } from 'zod';

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

/** The dataset and eval config are the heavy part, so they get a step of their own; where the
 *  results publish is a short, optional second step. */
const STEPS = ['Configuration', 'Run'];

const MAX_PARALLELISM = 16;

const EMPTY_PROMPT_MESSAGE = 'Select at least one field, or write a prompt.';

/** Why the primary action is disabled. A reason already shown under its field is not repeated
 *  beside the action. */
type Blocker = { kind: 'field' } | { kind: 'hint'; hint: string };

const SHOWN_ON_FIELD: Blocker = { kind: 'field' };

const blockedWith = (hint: string): Blocker => ({ kind: 'hint', hint });

const submitEvaluationBaseSchema = z.object({
  agent: z.string().min(1, 'Agent is required'),
  judgeModel: z.string(),
  /** Whether the eval config is built here, picked from a fileset, or uploaded. */
  configKind: z.enum(CONFIG_KINDS),
  /** The prompt sent to the agent; ``null`` while it is still the generated one. */
  promptTemplate: z.string().nullable(),
  /** Columns the generated agent prompt sends, in the order they were checked. */
  promptColumns: z.array(z.string()),
  /** Columns the generated judge prompt sends ahead of the agent's response. */
  judgeColumns: z.array(z.string()),
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
  ...runTargetSchemaShape,
});

type SubmitEvaluationFormData = z.infer<typeof submitEvaluationBaseSchema>;

const CONFIGURATION_FIELDS = ['dataset', 'promptTemplate', 'body', 'judgeModel'] as const;

const isUnusablePrompt = (prompt: string | null) => prompt !== null && hasMalformedBraces(prompt);

const makeSubmitEvaluationSchema = (requiresJudgeModel: () => boolean) =>
  submitEvaluationBaseSchema.superRefine((data, ctx) => {
    const issue = (path: (string | number)[], message: string) =>
      ctx.addIssue({ code: z.ZodIssueCode.custom, message, path });
    if (requiresJudgeModel() && !data.judgeModel) {
      issue(['judgeModel'], 'Select a model to override the config');
    }
    if (!data.dataset) issue(['dataset'], 'Select an input file.');
    if (data.configKind === CONFIG_KIND_FORM) {
      const promptEmpty =
        data.promptTemplate === null
          ? data.promptColumns.length === 0
          : !data.promptTemplate.trim();
      if (promptEmpty) issue(['promptTemplate'], EMPTY_PROMPT_MESSAGE);
      if (isUnusablePrompt(data.promptTemplate)) issue(['promptTemplate'], 'Write a prompt.');
      if (isUnusablePrompt(data.body.judgePrompt))
        issue(['body', 'judgePrompt'], 'Write a prompt.');
      if (!data.body.judgeModel) issue(['body', 'judgeModel'], 'Select a judge model.');
      if (data.body.scores.length === 0) {
        issue(['body', 'scores'], 'Add at least one score for the judge to extract.');
      }
    }
    // Blank is fine (a name is generated); anything typed must survive sanitizing, and the
    // fileset derived from it is longer, so it can overflow where the name alone would not.
    if (!data.evaluationRecordName) return;
    const nameError =
      unsalvageableNameError(data.evaluationRecordName, 'Evaluation name') ??
      getEntityNameError(dataFilesetName(data.evaluationRecordName), 'Derived fileset name');
    if (nameError) issue(['evaluationRecordName'], nameError);
  });

const makeDefaultValues = (agent: string): SubmitEvaluationFormData => ({
  agent,
  judgeModel: '',
  configKind: CONFIG_KIND_FORM,
  promptTemplate: null,
  promptColumns: [],
  judgeColumns: [],
  dataset: null,
  body: { judgeModel: '', judgePrompt: null, scores: EVALUATION_FORM_DEFAULTS.body.scores },
  configFileset: '',
  configFile: '',
  parallelism: DEFAULT_PARALLELISM,
  ...RUN_TARGET_DEFAULTS,
});

/** The picked eval config, with its parsed spec once it validates. ``format`` follows the
 *  uploaded extension, not the detected syntax: a file named .yaml is stored as YAML even if
 *  its contents happen to be valid JSON, which is the mapping an author expects. */
interface ConfigPick {
  file: File;
  error?: string;
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

/** How a dataset ref is labelled on the Evaluation: its fileset, workspace-qualified when it
 *  lives elsewhere. */
const datasetLabel = (ref: string, workspace: string): string | undefined => {
  const location = parseFilesetLocation(ref, workspace);
  if (!location) return undefined;
  return location.workspace === workspace
    ? location.name
    : `${location.workspace}/${location.name}`;
};

interface AgentEvaluationWizardProps {
  workspace: string;
  /** The agent under evaluation. */
  agent: string;
}

/** Undo what a failed submit created, and return the error to raise. When a delete itself fails
 *  the returned error names what to remove by hand. */
const discardSeeded = async (
  workspace: string,
  seeded: { filesetName?: string; evaluationName?: string },
  cause: unknown
): Promise<unknown> => {
  const leftovers: string[] = [];
  if (seeded.evaluationName) {
    await deleteEvaluation(workspace, seeded.evaluationName).catch(() =>
      leftovers.push(`evaluation "${seeded.evaluationName}"`)
    );
  }
  if (seeded.filesetName) {
    await filesDeleteFileset(workspace, seeded.filesetName).catch(() =>
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

/** The eval config, built or picked, and the dataset it reads in place. */
interface EvalInputs {
  spec: DatasetEvalSpec;
  /** Serialization the config was uploaded in; the stored file keeps it. */
  configFormat: EvalConfigFormat;
  /** A file, or a glob over Parquet shards. */
  datasetRef: string;
}

/** Create the run's fileset and store the final eval config in it: the config with the picked
 *  judge and the dataset ref baked in, so a later run can reuse it. */
const seedRunFileset = async (
  workspace: string,
  filesetName: string,
  judgeModel: string,
  inputs: EvalInputs
): Promise<DatasetEvalSpec> => {
  const spec: DatasetEvalSpec = {
    ...inputs.spec,
    dataset: inputs.datasetRef,
    metrics: judgeModel
      ? inputs.spec.metrics.map((m) => injectJudgeModel(m, judgeModel))
      : inputs.spec.metrics,
  };
  try {
    await filesCreateFileset(workspace, {
      name: filesetName,
      description: 'Agent Evaluation Config',
    });
  } catch (err) {
    if (isConflictError(err)) {
      throw new Error(
        `A fileset named "${filesetName}" already exists — choose a different evaluation name`
      );
    }
    throw err;
  }
  try {
    await filesUploadFile(
      workspace,
      filesetName,
      evalConfigFilename(inputs.configFormat),
      new Blob([serializeEvalConfig(spec, inputs.configFormat)], {
        type: inputs.configFormat === 'yaml' ? 'application/yaml' : 'application/json',
      })
    );
  } catch (uploadErr) {
    throw await discardSeeded(workspace, { filesetName }, uploadErr);
  }
  return spec;
};

export const AgentEvaluationWizard: FC<AgentEvaluationWizardProps> = ({ workspace, agent }) => {
  const toast = useToast();
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  // Ref keeps the override's required-ness current for the zod schema getter at validation time.
  const judgeRequiredRef = useRef(false);
  const [schema] = useState(() => makeSubmitEvaluationSchema(() => judgeRequiredRef.current));

  const [step, setStep] = useState(0);
  const [submitAttempted, setSubmitAttempted] = useState(false);
  const [configPick, setConfigPick] = useState<ConfigPick | null>(null);

  // Bumped whenever a pick is replaced, removed, or reset, so an async validation that is
  // still running when that happens knows to drop its result instead of committing it.
  const configToken = useRef(0);

  const methods = useForm<SubmitEvaluationFormData>({
    resolver: zodResolver(schema),
    defaultValues: makeDefaultValues(agent),
    mode: 'onSubmit',
    reValidateMode: 'onChange',
  });
  const { control, setValue, handleSubmit, clearErrors, trigger, setError, formState } = methods;
  const { errors } = formState;

  const configKind = useWatch({ control, name: 'configKind' });
  const isFormConfig = configKind === CONFIG_KIND_FORM;
  const [dataset, promptTemplate, judgePrompt, promptColumns, judgeColumns] = useWatch({
    control,
    name: ['dataset', 'promptTemplate', 'body.judgePrompt', 'promptColumns', 'judgeColumns'],
  });
  const datasetLocation = dataset ? parseFilesetLocation(dataset) : null;
  // Reading the sibling shards is opted into per file: another pick may share a folder with
  // files that are not part of it, such as a test split beside a train split.
  const [allShardsFor, setAllShardsFor] = useState<string | null>(null);
  const useAllShards = !!dataset && allShardsFor === dataset;
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

  // Parquet shards beside the picked file, read together through one glob when the box is ticked.
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
    !isFormConfig && pickedSpec && preview.row
      ? configFitIssues(pickedSpec, columns)
      : NO_FIT_ISSUES;
  const generatedAgentPrompt = defaultAgentPrompt(
    promptColumns.filter((column) => columns.includes(column))
  );
  const generatedJudgePrompt = defaultJudgePrompt(
    judgeColumns.filter((column) => columns.includes(column))
  );
  const agentPrompt = promptTemplate ?? generatedAgentPrompt;
  const checksPrompts = isFormConfig && !!preview.row;
  const promptIssues =
    checksPrompts && agentPrompt.trim() ? promptFitIssues(agentPrompt, columns) : NO_FIT_ISSUES;
  const judgePromptIssues = checksPrompts
    ? promptFitIssues(judgePrompt ?? generatedJudgePrompt, columns, true)
    : NO_FIT_ISSUES;

  const { status: recordNameStatus } = useRunNameCheck(workspace, control);

  const configError =
    activeConfigPick?.error ??
    (isConfigFromFileset ? configFilesetFile.error?.message : undefined) ??
    (submitAttempted && !activeConfigPick && !isLoadingFilesetConfig
      ? 'Select an eval configuration'
      : undefined);

  // Mirrors injectJudgeModel's own guard, so the picker is shown exactly when a metric would
  // have a model written into it — not just for the llm-judge type. Every match is collected,
  // not just the first: the override rewrites all of them, so all of them must be checked.
  const judgeMetrics =
    activeConfigPick?.spec?.metrics.filter(
      (metric) => metric.metric_type === 'llm-judge' || 'model' in metric.payload.metric
    ) ?? [];

  const isLlmJudge = !isFormConfig && judgeMetrics.length > 0;

  // A different query from the one behind the dropdown (lazy, paginated for search); this is the
  // only complete workspace list, so it decides what counts as valid.
  const { data: judgeModels, isLoading: isJudgeModelsLoading } = useJudgeModels({});

  const workspaceModelUrns = new Set<string>(
    judgeModels
      ?.map((model) => getURNFromNamedEntityRef(model))
      .filter((urn): urn is NonNullable<typeof urn> => urn !== undefined) ?? []
  );

  // Nothing to validate, but it still cannot run — so it requires the override just the same.
  const hasModellessJudge =
    isLlmJudge && judgeMetrics.some((metric) => typeof metric.payload.metric.model !== 'string');

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
  // the pick is no longer current. A fresh config also means a fresh judge.
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
    clearErrors(['judgeModel', 'configFileset', 'configFile', 'promptTemplate', 'body']);
    clearConfigPick();
    setValue('configFileset', '');
    setValue('configFile', '');
    setValue('configKind', value as ConfigKind);
  };

  const buildInputs = (formData: SubmitEvaluationFormData): EvalInputs | null => {
    if (!datasetRef) return null;
    if (formData.configKind === CONFIG_KIND_FORM) {
      return {
        spec: buildAgentEvaluationSpec(
          formData.dataset ?? '',
          formData.promptTemplate ?? generatedAgentPrompt,
          { ...formData.body, judgePrompt: formData.body.judgePrompt ?? generatedJudgePrompt },
          workspace
        ),
        configFormat: 'json',
        datasetRef,
      };
    }
    return pickedSpec
      ? { spec: pickedSpec, configFormat: activeConfigPick?.format ?? 'json', datasetRef }
      : null;
  };

  const {
    mutateAsync: submitEvaluation,
    error: submitError,
    isPending,
  } = useMutation({
    mutationFn: async ({
      formData,
      inputs,
    }: {
      formData: SubmitEvaluationFormData;
      inputs: EvalInputs;
    }) => {
      const snapshot = await takeDatasetSnapshot(workspace, inputs.datasetRef).catch(
        () => undefined
      );
      if (snapshot === null) throw new Error(`The dataset ${inputs.datasetRef} no longer exists.`);
      const { experimentIds, name } = await resolveRunTarget(workspace, formData.agent, formData);
      const filesetName = dataFilesetName(name);
      const spec = await seedRunFileset(workspace, filesetName, formData.judgeModel, inputs);
      const metadata = runMetadata(
        filesetName,
        snapshot ? { [DATASET_SNAPSHOT_KEY]: JSON.stringify(snapshot) } : undefined
      );
      const seeded: { filesetName: string; evaluationName?: string } = { filesetName };
      try {
        await createRunEvaluation(workspace, {
          experimentIds,
          name,
          filesetName,
          datasetName: datasetLabel(inputs.datasetRef, workspace),
          metadata,
        }).catch((err: unknown) => {
          if (isConflictError(err)) {
            throw new Error(
              `An evaluation named "${name}" already exists — choose a different evaluation name`
            );
          }
          throw err;
        });
        seeded.evaluationName = name;

        const created = await evalsCreateEvaluateJob(
          workspace,
          buildDatasetEvalRequestBody(
            spec,
            {
              workspace,
              agent: formData.agent,
              filesetName,
              experimentName: name,
              evaluationId: name,
              parallelism: formData.parallelism,
            },
            null
          ) as EvaluateJobRequest
        );
        if (!created?.name) throw new Error('Submission did not return a job name');
        await recordRunJob(workspace, name, created.name, metadata);
        return { name: created.name };
      } catch (err) {
        throw await discardSeeded(workspace, seeded, err);
      }
    },
    onSuccess: ({ name }, { formData }) => {
      toast.success(`Evaluation "${name}" submitted`);
      void queryClient.invalidateQueries({ queryKey: ['evaluator-jobs', workspace] });
      navigate(getAgentEvaluationsTabRoute(workspace, bareName(formData.agent)), {
        replace: true,
      });
    },
  });

  const onLastStep = step === STEPS.length - 1;

  // Everything the current step waits on, so the button is disabled instead of swallowing the
  // click. Schema errors stay out: those only show once the step is submitted.
  const blocker = ((): Blocker | undefined => {
    if (!onLastStep) {
      if (isReadingDataset || isLoadingFilesetConfig) {
        return blockedWith('Reading the selected file...');
      }
      if (datasetError) return blockedWith(datasetError);
      // invalidModelRefs is empty while the list is in flight, so advancing inside that window
      // would skip the judge-model check entirely.
      if (isLlmJudge && isJudgeModelsLoading) return blockedWith('Loading judge models...');
      if (
        configIssues.errors.length > 0 ||
        promptIssues.errors.length > 0 ||
        judgePromptIssues.errors.length > 0
      ) {
        return SHOWN_ON_FIELD;
      }
      return undefined;
    }
    if (recordNameStatus === 'checking' || recordNameStatus === 'conflict') {
      return SHOWN_ON_FIELD;
    }
    return undefined;
  })();

  const isAgentPromptMissing = () => {
    if (!isFormConfig || agentPrompt.trim()) return false;
    setError('promptTemplate', { type: 'custom', message: EMPTY_PROMPT_MESSAGE });
    return true;
  };

  const goNext = async () => {
    setSubmitAttempted(true);
    const valid = await trigger([...CONFIGURATION_FIELDS]);
    if (!valid || (!isFormConfig && !pickedSpec) || isAgentPromptMissing()) return;
    setStep(step + 1);
  };

  const goBack = () => {
    clearErrors();
    setStep(step - 1);
  };

  const onSubmit: SubmitHandler<SubmitEvaluationFormData> = async (formData) => {
    // EvalWizard already drops clicks and Enter while blocked; this keeps a submit from ever
    // skipping a gate.
    if (blocker) return;
    const inputs = buildInputs(formData);
    if (!inputs) {
      setStep(0);
      return;
    }
    try {
      await submitEvaluation({ formData, inputs });
    } catch {
      // Error rendered via errorText prop.
    }
  };

  // Enter on the first step means "next", not "submit the half-filled form".
  const handleContinue = () => {
    if (blocker) return;
    if (!onLastStep) {
      void goNext();
      return;
    }
    void handleSubmit(onSubmit, (invalid) => {
      if (CONFIGURATION_FIELDS.some((field) => field in invalid)) setStep(0);
    })();
  };

  const errorMessage = submitErrorMessage(submitError);

  return (
    <EvalWizard
      heading="Run Agent Evaluation"
      headingDescription="Score your agent against a dataset."
      steps={STEPS}
      activeStep={step}
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
        step > 0 ? (
          <Button kind="secondary" onClick={goBack} disabled={isPending}>
            Back
          </Button>
        ) : null
      }
      cancelTo={getAgentEvaluationsTabRoute(workspace, agent)}
      continueLabel={onLastStep ? 'Run Evaluation' : 'Next'}
      continueLoading={isPending}
      canContinue={!blocker}
      blockedHint={blocker?.kind === 'hint' ? blocker.hint : undefined}
      onContinue={handleContinue}
    >
      <FormProvider {...methods}>
        <Stack gap="density-xl">
          {step === 0 && (
            <>
              <Text kind="body/regular/md">
                Pick a dataset and an eval config to score your agent with. Runs are evaluated by
                the NeMo Helix Evals built-in runner.{' '}
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

              <DatasetPanel
                slotLabel={<Text kind="label/bold/xl">Dataset File</Text>}
                slotHelp="Read from its fileset, not copied, so later changes to this file also change re-runs."
                acceptedFileTypes={DATASET_FILE_TYPES}
                showMapping={false}
                onFileSelected={() => {
                  setValue('promptColumns', []);
                  setValue('judgeColumns', []);
                }}
              >
                {shards ? (
                  <Checkbox
                    checked={useAllShards}
                    onCheckedChange={(checked) =>
                      setAllShardsFor(checked === true ? dataset : null)
                    }
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

                    {configIssues.errors.map((issue) => (
                      <Banner key={issue} kind="inline" status="error">
                        {issue}
                      </Banner>
                    ))}
                    {configIssues.warnings.map((issue) => (
                      <Banner key={issue} kind="inline" status="warning">
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
                    columnsName="promptColumns"
                    title="Agent Prompt"
                    titleKind="label/bold/lg"
                    description="Sent to the agent for every row. Select the fields it should see, and leave out expected answers."
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
                      columnsName="judgeColumns"
                      title="Judge Prompt"
                      titleKind="label/bold/md"
                      description="Sent to the judge with the agent's response for every row. Select the fields to check the response against, such as expected answers."
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
                    <code>{bareName(agent)}</code>. Any target in the config is replaced by this
                    agent.
                  </Text>
                  <Stack gap="density-sm">
                    {pickedSpec.metrics.map((metric, index) => {
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
            </>
          )}

          {step === 1 && (
            <>
              <Text kind="label/bold/xl">Results</Text>
              <RunTargetFields workspace={workspace} agent={agent} disabled={isPending} />
              <ControlledTextInput
                useControllerProps={{ control, name: 'parallelism' }}
                name="parallelism"
                label="Parallel Requests"
                type="number"
                formFieldProps={{
                  slotInfo:
                    'Rows sent to the agent at once. Lower it for a slow agent: a deployment that answers one request at a time fails rows that wait too long in its queue.',
                  slotError: errors.parallelism?.message,
                }}
              />
            </>
          )}
        </Stack>
      </FormProvider>
    </EvalWizard>
  );
};
