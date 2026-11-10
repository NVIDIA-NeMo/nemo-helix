// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { isConflictError } from '@nemo/common/src/api/common/utils';
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
} from '@nemo/sdk/generated/platform/files';
import {
  Anchor,
  Banner,
  FormField,
  SegmentedControl,
  Stack,
  Text,
  Upload,
} from '@nvidia/foundations-react-core';
import { useFilesetFile } from '@studio/api/files/useFilesetFile';
import {
  DATASET_FILE_ACCEPT,
  EVAL_CONFIG_FILE_ACCEPT,
} from '@studio/components/evaluation/AgentEvaluationWizard/consts';
import { FilesetConfigPicker } from '@studio/components/evaluation/AgentEvaluationWizard/FilesetConfigPicker';
import { FilesetDatasetPicker } from '@studio/components/evaluation/AgentEvaluationWizard/FilesetDatasetPicker';
import {
  type InspectedDataset,
  inspectDatasetFiles,
  type StoredDatasetFile,
} from '@studio/components/evaluation/AgentEvaluationWizard/inspectDatasetFile';
import {
  resolveRunTarget,
  RUN_TARGET_DEFAULTS,
  runTargetSchemaShape,
  useRunNameCheck,
} from '@studio/components/evaluation/AgentEvaluationWizard/runTarget';
import { RunTargetFields } from '@studio/components/evaluation/AgentEvaluationWizard/RunTargetFields';
import { EvalWizard } from '@studio/components/evaluation/EvalWizard';
import {
  createRunEvaluation,
  EVAL_CONFIG_FILESET_KEY,
  evalConfigFilename,
  recordRunJob,
} from '@studio/components/evaluation/experimentEvalConfig';
import { JudgeModelSelect } from '@studio/components/evaluation/JudgeModelSelect';
import {
  submitErrorMessage,
  unsalvageableNameError,
} from '@studio/components/evaluation/shared/entityNameField';
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
import { type FC, useRef, useState } from 'react';
import { FormProvider, type SubmitHandler, useForm, useWatch } from 'react-hook-form';
import { useNavigate } from 'react-router';
import { z } from 'zod';

const FILE_SOURCE_UPLOAD = 'upload';
const FILE_SOURCE_FILESET = 'fileset';

type FileSource = typeof FILE_SOURCE_UPLOAD | typeof FILE_SOURCE_FILESET;

const FILE_SOURCE_ITEMS = [
  { value: FILE_SOURCE_UPLOAD, children: 'Upload a file' },
  { value: FILE_SOURCE_FILESET, children: 'Choose from a fileset' },
];

const MAX_PARALLELISM = 16;

/** Why Run Evaluation is disabled. A reason already shown under its field is not repeated
 *  beside the action. */
type Blocker = { kind: 'field' } | { kind: 'hint'; hint: string };

const SHOWN_ON_FIELD: Blocker = { kind: 'field' };

const blockedWith = (hint: string): Blocker => ({ kind: 'hint', hint });

const submitEvaluationBaseSchema = z.object({
  agent: z.string().min(1, 'Agent is required'),
  judgeModel: z.string(),
  datasetFileset: z.string(),
  datasetFile: z.string(),
  datasetBatchGlob: z.string(),
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

const makeSubmitEvaluationSchema = (requiresJudgeModel: () => boolean) =>
  submitEvaluationBaseSchema.superRefine((data, ctx) => {
    if (requiresJudgeModel() && !data.judgeModel) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'Select a model to override the config',
        path: ['judgeModel'],
      });
    }
    // Blank is fine (a name is generated); anything typed must survive sanitizing, and the
    // fileset derived from it is longer, so it can overflow where the name alone would not.
    if (!data.evaluationRecordName) return;
    const nameError =
      unsalvageableNameError(data.evaluationRecordName, 'Evaluation name') ??
      getEntityNameError(dataFilesetName(data.evaluationRecordName), 'Derived fileset name');
    if (nameError) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: nameError,
        path: ['evaluationRecordName'],
      });
    }
  });

const makeDefaultValues = (agent: string): SubmitEvaluationFormData => ({
  agent,
  judgeModel: '',
  datasetFileset: '',
  datasetFile: '',
  datasetBatchGlob: '',
  configFileset: '',
  configFile: '',
  parallelism: DEFAULT_PARALLELISM,
  ...RUN_TARGET_DEFAULTS,
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

/** The dataset and eval config picked for the run, already validated. */
interface UploadedEvalInputs {
  /** Copied into the run's fileset; empty when the dataset is read in place. */
  datasetFiles: StoredDatasetFile[];
  /** Where the eval config reads the dataset within the run's fileset. */
  datasetPath: string;
  /** Read the dataset in place from this ref instead of from the run's fileset. */
  datasetRef?: string;
  spec: DatasetEvalSpec;
  /** Serialization the config was uploaded in; the stored file keeps it. */
  configFormat: EvalConfigFormat;
}

/** Create the run's fileset and store the dataset and the final eval config in it: the uploaded
 *  config with the picked judge and a ref to the dataset baked in, so a later run can reuse it. */
const seedRunFileset = async (
  workspace: string,
  filesetName: string,
  judgeModel: string,
  uploads: UploadedEvalInputs
): Promise<DatasetEvalSpec> => {
  const spec: DatasetEvalSpec = {
    ...uploads.spec,
    dataset: uploads.datasetRef ?? `${workspace}/${filesetName}#${uploads.datasetPath}`,
    metrics: judgeModel
      ? uploads.spec.metrics.map((m) => injectJudgeModel(m, judgeModel))
      : uploads.spec.metrics,
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
    await Promise.all(
      uploads.datasetFiles.map(({ name: path, file }) =>
        filesUploadFile(workspace, filesetName, path, file)
      )
    );
    await filesUploadFile(
      workspace,
      filesetName,
      evalConfigFilename(uploads.configFormat),
      new Blob([serializeEvalConfig(spec, uploads.configFormat)], {
        type: uploads.configFormat === 'yaml' ? 'application/yaml' : 'application/json',
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

  const [submitAttempted, setSubmitAttempted] = useState(false);
  const [datasetPick, setDatasetPick] = useState<InspectedDataset | null>(null);
  const [datasetSource, setDatasetSource] = useState<FileSource>(FILE_SOURCE_UPLOAD);
  const [configPick, setConfigPick] = useState<ConfigPick | null>(null);
  const [configSource, setConfigSource] = useState<FileSource>(FILE_SOURCE_UPLOAD);

  // Bumped whenever a pick is replaced, removed, or reset, so an async validation that is
  // still running when that happens knows to drop its result instead of committing it.
  const datasetToken = useRef(0);
  const configToken = useRef(0);

  const methods = useForm<SubmitEvaluationFormData>({
    resolver: zodResolver(schema),
    defaultValues: makeDefaultValues(agent),
    mode: 'onSubmit',
    reValidateMode: 'onChange',
  });
  const { control, setValue, handleSubmit, formState } = methods;
  const { errors } = formState;

  const datasetFileset = useWatch({ control, name: 'datasetFileset' });
  const datasetFilePath = useWatch({ control, name: 'datasetFile' });
  const datasetBatchGlob = useWatch({ control, name: 'datasetBatchGlob' });
  const isFilesetSource = datasetSource === FILE_SOURCE_FILESET;
  // Downloaded raw rather than through useDatasetFileContent, which hands Parquet back as JSONL
  // text; the run stores the file as-is so the evaluator reads it with its own loader.
  const filesetFile = useFilesetFile({
    workspace,
    fileset: datasetFileset,
    path: datasetFilePath,
    enabled: isFilesetSource,
  });
  const downloadedFile = filesetFile.data;
  const filesetInspection = useQuery({
    queryKey: [
      'evaluation-dataset-pick',
      workspace,
      datasetFileset,
      datasetFilePath,
      filesetFile.dataUpdatedAt,
    ],
    queryFn: downloadedFile ? () => inspectDatasetFiles([downloadedFile]) : skipToken,
    gcTime: 0,
  });
  const isLoadingFilesetFile = filesetFile.isFetching || filesetInspection.isFetching;
  const filesetPick = isFilesetSource ? (filesetInspection.data ?? null) : null;
  const activeDatasetPick = isFilesetSource ? filesetPick : datasetPick;

  const configFileset = useWatch({ control, name: 'configFileset' });
  const configFilePath = useWatch({ control, name: 'configFile' });
  const isConfigFromFileset = configSource === FILE_SOURCE_FILESET;
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

  const uploads: UploadedEvalInputs | null =
    activeDatasetPick?.storedName &&
    !activeDatasetPick.error &&
    activeConfigPick?.spec &&
    !activeConfigPick.error
      ? {
          ...(isFilesetSource && datasetBatchGlob
            ? { datasetFiles: [], datasetRef: `${workspace}/${datasetFileset}#${datasetBatchGlob}` }
            : { datasetFiles: activeDatasetPick.stored }),
          datasetPath: activeDatasetPick.storedName,
          spec: activeConfigPick.spec,
          configFormat: activeConfigPick.format ?? 'json',
        }
      : null;

  const { status: recordNameStatus } = useRunNameCheck(workspace, control);

  const datasetError =
    activeDatasetPick?.error ??
    (isFilesetSource ? filesetFile.error?.message : undefined) ??
    (submitAttempted && !activeDatasetPick && !isLoadingFilesetFile ? 'Add a dataset' : undefined);
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

  const isLlmJudge = judgeMetrics.length > 0;

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

  const clearDatasetPick = () => {
    datasetToken.current += 1;
    setDatasetPick(null);
  };

  const clearConfigPick = () => {
    configToken.current += 1;
    setConfigPick(null);
  };

  // Both handlers clear first: validation is async, and leaving the previous pick in place
  // would keep the submit gate open against a file the user has already replaced. Each also
  // drops its own result if the pick it belongs to is no longer current — a slow read of a
  // replaced file would otherwise land after a faster one and submit a file the card no
  // longer shows. The tokens are per input: one shared counter would let a config pick
  // cancel an in-flight dataset read, stranding the form with no pick and no error.
  // Removing a dataset file calls onValueChange with the files left, so this handles removal too.
  const handleDatasetPicked = async (items: readonly { file: File }[]) => {
    clearDatasetPick();
    if (items.length === 0) return;
    const token = datasetToken.current;
    const inspected = await inspectDatasetFiles(items.map(({ file }) => file));
    if (token !== datasetToken.current) return;
    setDatasetPick(inspected);
  };

  // A fresh config also means a fresh judge, or the previous file's judge would carry over.
  const handleConfigPicked = async (item?: { file: File }) => {
    setValue('judgeModel', '');
    clearConfigPick();
    if (!item?.file) return;
    const token = configToken.current;
    const pick = await readConfigPick(item.file);
    if (token !== configToken.current) return;
    setConfigPick(pick);
  };

  const handleConfigSourceChanged = (value: string) => {
    setValue('judgeModel', '');
    clearConfigPick();
    setValue('configFileset', '');
    setValue('configFile', '');
    setConfigSource(value as FileSource);
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
      inputs: UploadedEvalInputs;
    }) => {
      const { experimentIds, name } = await resolveRunTarget(workspace, formData.agent, formData);
      const filesetName = dataFilesetName(name);
      const spec = await seedRunFileset(workspace, filesetName, formData.judgeModel, inputs);
      const seeded: { filesetName: string; evaluationName?: string } = { filesetName };
      try {
        await createRunEvaluation(workspace, { experimentIds, name, filesetName }).catch(
          (err: unknown) => {
            if (isConflictError(err)) {
              throw new Error(
                `An evaluation named "${name}" already exists — choose a different evaluation name`
              );
            }
            throw err;
          }
        );
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
        await recordRunJob(workspace, name, created.name, {
          [EVAL_CONFIG_FILESET_KEY]: filesetName,
        });
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

  // Everything Run Evaluation waits on, so the button is disabled instead of swallowing the
  // click. Schema errors stay out: those only show once a submit is tried.
  const blocker = ((): Blocker | undefined => {
    if (isLoadingFilesetFile || isLoadingFilesetConfig) {
      return blockedWith('Reading the selected file...');
    }
    // invalidModelRefs is empty while the list is in flight, so submitting inside that window
    // would skip the judge-model check entirely.
    if (isLlmJudge && isJudgeModelsLoading) {
      return blockedWith('Loading judge models...');
    }
    if (recordNameStatus === 'checking' || recordNameStatus === 'conflict') {
      return SHOWN_ON_FIELD;
    }
    return undefined;
  })();

  const onSubmit: SubmitHandler<SubmitEvaluationFormData> = async (formData) => {
    // EvalWizard already drops clicks and Enter while blocked; this keeps a submit from ever
    // skipping a gate. Missing files are reported on their fields once submitAttempted is set.
    if (blocker || !uploads) return;
    try {
      await submitEvaluation({ formData, inputs: uploads });
    } catch {
      // Error rendered via errorText prop.
    }
  };

  const handleContinue = () => {
    setSubmitAttempted(true);
    void handleSubmit(onSubmit)();
  };

  const errorMessage = submitErrorMessage(submitError);

  return (
    <EvalWizard
      heading="Run Agent Evaluation"
      headingDescription="Score your agent against a dataset."
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
      cancelTo={getAgentEvaluationsTabRoute(workspace, agent)}
      continueLabel="Run Evaluation"
      continueLoading={isPending}
      canContinue={!blocker}
      blockedHint={blocker?.kind === 'hint' ? blocker.hint : undefined}
      onContinue={handleContinue}
    >
      <FormProvider {...methods}>
        <Stack gap="density-xl">
          <Text kind="body/regular/md">
            Pick a dataset and an eval config to score your agent with. Runs are evaluated by the
            NeMo Helix Evals built-in runner.{' '}
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

          <Text kind="label/bold/xl">Dataset File</Text>
          <SegmentedControl
            aria-label="Dataset source"
            size="tiny"
            className="w-full"
            value={datasetSource}
            onValueChange={(value) => {
              clearDatasetPick();
              setValue('datasetFileset', '');
              setValue('datasetFile', '');
              setValue('datasetBatchGlob', '');
              setDatasetSource(value as FileSource);
            }}
            items={FILE_SOURCE_ITEMS}
          />

          {isFilesetSource ? (
            <FilesetDatasetPicker<SubmitEvaluationFormData>
              workspace={workspace}
              control={control}
              filesetName="datasetFileset"
              fileName="datasetFile"
              batchGlobName="datasetBatchGlob"
              disabled={isPending}
              loading={isLoadingFilesetFile}
              error={datasetError}
            />
          ) : (
            <Upload
              attributes={{ UploadInputElement: { 'aria-label': 'Dataset file' } }}
              accept={DATASET_FILE_ACCEPT}
              multiple
              onValueChange={handleDatasetPicked}
              status={datasetError ? 'error' : undefined}
              renderInput={(slotInput) => (
                <FormField
                  name="dataset"
                  slotHelp="JSONL, a JSON array of objects, CSV, or Parquet. Select several Parquet files to combine them."
                  slotError={datasetError}
                  status={datasetError ? 'error' : undefined}
                >
                  {datasetPick ? null : slotInput}
                </FormField>
              )}
            />
          )}

          <Text kind="label/bold/xl">Eval Configuration File</Text>
          <SegmentedControl
            aria-label="Evaluator config source"
            size="tiny"
            className="w-full"
            value={configSource}
            onValueChange={handleConfigSourceChanged}
            items={FILE_SOURCE_ITEMS}
          />

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
              attributes={{ UploadInputElement: { 'aria-label': 'Eval configuration file' } }}
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

          {isLlmJudge && (
            <JudgeModelSelect<SubmitEvaluationFormData>
              formFieldName="judgeModel"
              slotLabel={
                judgeRequired ? 'Override All LLM Models' : 'Override All LLM Models (Optional)'
              }
              slotError={invalidModelsError}
            />
          )}

          <Text kind="label/bold/xl">Results</Text>
          <RunTargetFields workspace={workspace} agent={agent} disabled={isPending} />

          <ControlledTextInput
            useControllerProps={{ control, name: 'parallelism' }}
            name="parallelism"
            label="Parallel requests"
            type="number"
            formFieldProps={{
              slotInfo:
                'Rows sent to the agent at once. Lower it for a slow agent: a deployment that answers one request at a time fails rows that wait too long in its queue.',
              slotError: errors.parallelism?.message,
            }}
          />
        </Stack>
      </FormProvider>
    </EvalWizard>
  );
};
