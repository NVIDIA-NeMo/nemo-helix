// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal, type FormModalProps } from '@nemo/common/src/components/FormModal';
import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { DEFAULT_DEBOUNCE_MS } from '@nemo/common/src/constants';
import { getURNFromNamedEntityRef } from '@nemo/common/src/namedEntity';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { getEntityNameError, toValidEntityName } from '@nemo/common/src/utils/entityName';
import { useAgentsListAgents } from '@nemo/sdk/generated/agents/agents';
import { evaluatorCreateEvaluateJob } from '@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes';
import type {
  AgentEvaluateJobRequest,
  EvaluateJobRequest,
} from '@nemo/sdk/generated/evaluator/schema';
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
} from '@nemo/sdk/generated/platform/files';
import {
  Anchor,
  Button,
  Flex,
  FormField,
  RadioGroup,
  Stack,
  Stepper,
  Text,
  Tooltip,
  Upload,
} from '@nvidia/foundations-react-core';
import { submitAgentEvalJob } from '@studio/api/evaluation/agent-evaluations';
import { isConflictError } from '@studio/api/evaluation/eval-config-fileset';
import {
  createRunEvaluation,
  evalConfigFilename,
  evaluationConfigError,
  evaluationFilesetName,
  findEvalConfigFile,
} from '@studio/components/evaluation/experimentEvalConfig';
import { JudgeModelSelect } from '@studio/components/evaluation/JudgeModelSelect';
import '@studio/components/evaluation/SubmitEvaluationModal.css';
import {
  entityNameField,
  nameCheckStatus,
  nameFieldSlots,
  submitErrorMessage,
  unsalvageableNameError,
} from '@studio/components/evaluation/shared/entityNameField';
import { EvaluationSourceSelect } from '@studio/components/evaluation/shared/EvaluationSourceSelect';
import {
  EXPERIMENT_SETTINGS_DEFAULTS,
  experimentSettingsPayload,
  experimentSettingsSchemaShape,
} from '@studio/components/evaluation/shared/experimentSettings';
import { ExperimentSettingsFields } from '@studio/components/evaluation/shared/ExperimentSettingsFields';
import { useEvaluationSources } from '@studio/components/evaluation/shared/useEvaluationSources';
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
import {
  type EvaluationMode,
  isLastStep,
  nextStep,
  previousStep,
  stepHeading,
  stepIndex,
  stepsFor,
  type WizardStep,
} from '@studio/components/evaluation/wizardSteps';
import { LINK_DOCS_STUDIO_EXPERIMENTS, LINK_EVAL_DOCS } from '@studio/constants/links';
import { useJudgeModels } from '@studio/hooks/evaluation/useJudgeModels';
import { getAgentEvaluationsTabRoute } from '@studio/routes/utils';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Info } from 'lucide-react';
import { type FC, useCallback, useEffect, useRef, useState } from 'react';
import { FormProvider, type SubmitHandler, useForm, useWatch } from 'react-hook-form';
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
              <Info size={14} aria-label="Why is this option disabled?" />
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

/** Stem the dataset is stored under in the run's fileset; the extension follows its content. */
const DATASET_BASENAME = 'dataset';

const MAX_PARALLELISM = 16;

const NO_EVALUATIONS_MESSAGE =
  'No evaluations with a reusable eval config yet. Go back and create an experiment instead — its run is re-runnable from here afterwards.';

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
  parallelism: z.coerce
    .number()
    .int('Use a whole number')
    .min(1, 'At least 1')
    .max(MAX_PARALLELISM, `At most ${MAX_PARALLELISM}`),
  ...experimentSettingsSchemaShape,
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
  parallelism: DEFAULT_PARALLELISM,
  ...EXPERIMENT_SETTINGS_DEFAULTS,
});

/** A file the user picked, plus why it was rejected when it was. */
interface FilePick {
  file: File;
  error?: string;
}

/** The picked dataset, with the name it will be stored under once it validates. */
interface DatasetPick extends FilePick {
  storedName?: string;
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

const isRecord = (value: unknown): boolean =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** Validate the dataset and settle the name it is stored under. The evaluator's loader takes
 *  ``.json`` and ``.jsonl`` interchangeably, sniffing a leading ``[`` to tell an array from
 *  line-delimited records — so the name follows the detected content, not the uploaded
 *  extension, and the dataset ref written into the config matches.
 *
 *  Every record must be an object: the evaluator turns each one into a row keyed by its own
 *  fields, and a file of scalars only fails once the job is running. Parsed here rather than
 *  through ``validateFileFormat``, which does not check record shape and would mean reading a
 *  large dataset into memory twice. */
const inspectDatasetFile = async (file: File): Promise<Omit<DatasetPick, 'file'>> => {
  const text = (await file.text()).trim();
  if (!text) return { error: 'File is empty' };

  let records: unknown[];
  let format: 'json' | 'jsonl';
  try {
    const parsed: unknown = JSON.parse(text);
    records = Array.isArray(parsed) ? parsed : [parsed];
    format = 'json';
  } catch {
    try {
      records = text.split('\n').flatMap((line) => (line.trim() ? [JSON.parse(line)] : []));
      format = 'jsonl';
    } catch {
      return { error: 'File is not valid JSON or JSONL' };
    }
  }

  if (records.length === 0) return { error: 'File contains no data' };
  if (!records.every(isRecord)) {
    return { error: 'Every dataset record must be a JSON object.' };
  }
  return { storedName: `${DATASET_BASENAME}.${format}` };
};

interface SubmitEvaluationModalProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  workspace: string;
  /** When provided, pre-fills + locks the agent selector. */
  agent?: string;
  /** Name of an evaluation to start from. Opens the wizard on its last step with that run already
   *  chosen, which is how "New evaluation from this configuration" hands a row's config over. */
  sourceEvaluation?: string;
  /** Called after a successful submission with the new job's name. */
  onSubmitted?: (jobName: string) => void;
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

/** The two files uploaded on the "new experiment" path, already validated. */
interface UploadedEvalInputs {
  dataset: File;
  datasetName: string;
  spec: DatasetEvalSpec;
  /** Serialization the config was uploaded in; the stored file keeps it. */
  configFormat: EvalConfigFormat;
}

/** Resolves the persisted yardstick spec for this submission. On the "new experiment" path it
 *  takes the uploaded config, bakes in the picked judge and a ref to the dataset that is
 *  about to be uploaded beside it, and seeds both into a new fileset; on the re-run path it
 *  reads the saved spec back verbatim (no re-bake, no judge re-pick). */
const loadPersistedSpec = async (
  workspace: string,
  formData: SubmitEvaluationFormData,
  configFileset: string | null,
  uploads: UploadedEvalInputs | null
): Promise<EvalSpec> => {
  if (formData.mode === MODE_DEFAULT) {
    if (!uploads) throw new Error('Upload a dataset and an eval config before submitting');
    const signal = new AbortController().signal;
    const name = filesetNameForExperiment(formData.newName);
    const judgeModel = formData.judgeModel || null;
    const spec: DatasetEvalSpec = {
      ...uploads.spec,
      dataset: `${workspace}/${name}#${uploads.datasetName}`,
      metrics: judgeModel
        ? uploads.spec.metrics.map((m) => injectJudgeModel(m, judgeModel))
        : uploads.spec.metrics,
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
      await filesUploadFile(workspace, name, uploads.datasetName, uploads.dataset, signal);
      await filesUploadFile(
        workspace,
        name,
        evalConfigFilename(uploads.configFormat),
        new Blob([serializeEvalConfig(spec, uploads.configFormat)], {
          type: uploads.configFormat === 'yaml' ? 'application/yaml' : 'application/json',
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

export const SubmitEvaluationModal: FC<SubmitEvaluationModalProps> = ({
  open,
  onClose,
  workspace,
  agent: agentProp,
  sourceEvaluation,
  onSubmitted,
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
  // A handed-in source only skips the first step when its agent came with it; the picker is
  // agent-scoped, so without one the evaluation step has nothing to show.
  const startingStep = useCallback(
    (): WizardStep => (sourceEvaluation && agentProp ? 'evaluation' : 'start'),
    [sourceEvaluation, agentProp]
  );
  const [step, setStep] = useState<WizardStep>(startingStep);
  const [submitAttempted, setSubmitAttempted] = useState(false);
  const [datasetPick, setDatasetPick] = useState<DatasetPick | null>(null);
  const [configPick, setConfigPick] = useState<ConfigPick | null>(null);

  // Bumped whenever a pick is replaced, removed, or reset, so an async validation that is
  // still running when that happens knows to drop its result instead of committing it.
  const datasetToken = useRef(0);
  const configToken = useRef(0);

  const { data: agentsResponse, isLoading: isAgentsLoading } = useAgentsListAgents(
    workspace,
    undefined,
    { query: { enabled: open && !agentProp } }
  );
  const agents = agentsResponse?.data ?? [];

  const methods = useForm<SubmitEvaluationFormData>({
    resolver: zodResolver(schema),
    defaultValues: makeDefaultValues(agentProp, sourceEvaluation),
    mode: 'onSubmit',
    reValidateMode: 'onChange',
  });
  const { control, reset: resetForm, setValue, handleSubmit, clearErrors, formState } = methods;
  const { errors } = formState;

  const mode = useWatch({ control, name: 'mode' }) as EvaluationMode;
  const selectedAgent = useWatch({ control, name: 'agent' });
  const steps = stepsFor(mode);

  const agentFieldError = errors.agent?.message;
  const evaluationName = useWatch({ control, name: 'evaluationName' });

  // Every reusable evaluation for this agent, resolved to its experiment. One list feeds the
  // picker, its section headings, and the lookup that resolves the chosen run.
  const sources = useEvaluationSources({
    workspace,
    agent: selectedAgent || undefined,
    enabled: open && !!selectedAgent,
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
    if (!open || !selectedAgent || sources.isLoading || modeDefaultApplied.current) return;
    modeDefaultApplied.current = true;
    if (sources.isEmpty) setValue('mode', MODE_DEFAULT);
  }, [open, selectedAgent, sources.isLoading, sources.isEmpty, setValue]);

  // Nothing is preselected here. A step whose whole job is "choose the run to re-run" should not
  // answer itself — Next stays disabled until the user picks, and stepBlocker says why. (A run
  // handed in by "New evaluation from this configuration" arrives in the form's default values,
  // which is a choice already made rather than one guessed at.)

  const { data: evaluationConfigIssue, isFetching: isValidatingEvaluation } = useQuery({
    queryKey: ['evaluation-eval-config', workspace, evaluationName],
    queryFn: ({ signal }) =>
      selectedEvaluation ? evaluationConfigError(workspace, selectedEvaluation, signal) : null,
    enabled: open && mode === MODE_EXPERIMENT && !!selectedEvaluation,
  });

  const evaluationFileset = selectedEvaluation ? evaluationFilesetName(selectedEvaluation) : null;
  const evaluationFieldError = errors.evaluationName?.message ?? evaluationConfigIssue ?? undefined;

  const uploads: UploadedEvalInputs | null =
    datasetPick?.storedName && !datasetPick.error && configPick?.spec && !configPick.error
      ? {
          dataset: datasetPick.file,
          datasetName: datasetPick.storedName,
          spec: configPick.spec,
          configFormat: configPick.format ?? 'json',
        }
      : null;

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
    { query: { enabled: open && isCreateMode && !!debouncedExperimentName } }
  );
  const recordConflictQuery = useListEvaluations(
    workspace,
    { page_size: 1, filter: { name: debouncedRecordName } },
    { query: { enabled: open && !!debouncedRecordName } }
  );

  const datasetError =
    datasetPick?.error ?? (submitAttempted && !datasetPick ? 'Add a dataset' : undefined);
  const configError =
    configPick?.error ??
    (submitAttempted && !configPick ? 'Select an evaluator config' : undefined);

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

  // Only a conflict blocks here. The schema's own errors already stop handleSubmit; a conflict
  // lives outside formState, so without this the request would fire and come back a 409.
  const hasNameConflict =
    recordNameStatus === 'conflict' || (isCreateMode && experimentNameStatus === 'conflict');

  // Mirrors injectJudgeModel's own guard, so the picker is shown exactly when a metric would
  // have a model written into it — not just for the llm-judge type. Every match is collected,
  // not just the first: the override rewrites all of them, so all of them must be checked.
  const judgeMetrics =
    configPick?.spec?.metrics.filter(
      (metric) => metric.metric_type === 'llm-judge' || 'model' in metric.payload.metric
    ) ?? [];

  const isLlmJudge = mode === MODE_DEFAULT && judgeMetrics.length > 0;

  // A different query from the one behind the dropdown (lazy, paginated for search); this is the
  // only complete workspace list, so it decides what counts as valid.
  const { data: judgeModels, isLoading: isJudgeModelsLoading } = useJudgeModels({ enabled: open });

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
  // Removing a file calls onValueChange with no item, so the argument is optional.
  const handleDatasetPicked = async (item?: { file: File }) => {
    clearDatasetPick();
    if (!item?.file) return;
    const token = datasetToken.current;
    const inspected = await inspectDatasetFile(item.file);
    if (token !== datasetToken.current) return;
    setDatasetPick({ file: item.file, ...inspected });
  };

  // A fresh config also means a fresh judge: the preselect effect above bails once judgeModel
  // is set, so leaving it would keep the previous file's judge.
  const handleConfigPicked = async (item?: { file: File }) => {
    setValue('judgeModel', '');
    clearConfigPick();
    if (!item?.file) return;
    const token = configToken.current;
    let pick: ConfigPick;
    try {
      pick = {
        file: item.file,
        spec: parseUploadedDatasetConfig(await item.file.text()),
        format: configFormatForFile(item.file.name),
      };
    } catch (err) {
      pick = {
        file: item.file,
        error: err instanceof Error ? err.message : 'Could not read the file',
      };
    }
    if (token !== configToken.current) return;
    setConfigPick(pick);
  };

  const {
    mutateAsync: submitEvaluation,
    error: submitError,
    isPending,
    reset: resetMutation,
  } = useMutation({
    mutationFn: async (formData: SubmitEvaluationFormData) => {
      const spec = await loadPersistedSpec(workspace, formData, evaluationFileset, uploads);

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
          ? await evaluatorCreateEvaluateJob(
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
      onSubmitted?.(name);
      resetAndClose();
      navigate(getAgentEvaluationsTabRoute(workspace, bareName(formData.agent)));
    },
  });

  useEffect(() => {
    if (open) return;
    resetForm(makeDefaultValues(agentProp, sourceEvaluation));
    setStep(startingStep());
    suggestedRecordName.current = '';
    seededForSource.current = null;
    modeDefaultApplied.current = false;
    datasetToken.current += 1;
    configToken.current += 1;
    setDatasetPick(null);
    setConfigPick(null);
    setSubmitAttempted(false);
  }, [open, agentProp, sourceEvaluation, resetForm, startingStep]);

  // Seed the locked agent and the handed-in source on open. A blanket reset here would clobber
  // the judge-model preselect above, which runs earlier in effect order.
  useEffect(() => {
    if (!open) return;
    if (agentProp) setValue('agent', agentProp);
    if (sourceEvaluation) {
      setValue('mode', MODE_EXPERIMENT);
      setValue('evaluationName', sourceEvaluation);
    }
  }, [open, agentProp, sourceEvaluation, setValue]);

  const resetAndClose = () => {
    resetMutation();
    resetForm(makeDefaultValues(agentProp, sourceEvaluation));
    setStep(startingStep());
    suggestedRecordName.current = '';
    seededForSource.current = null;
    modeDefaultApplied.current = false;
    clearDatasetPick();
    clearConfigPick();
    setSubmitAttempted(false);
    onClose();
  };

  // What the current step still needs before Next means anything. Kept separate from the zod
  // schema: the schema judges the whole submission, and a step must only answer for its own
  // fields — otherwise step one would refuse to advance over a field two steps away.
  const stepBlocker = (): string | undefined => {
    if (step === 'start') return selectedAgent ? undefined : 'Pick an agent to evaluate.';
    if (step === 'experiment') {
      if (!experimentPreview) return 'Name the experiment to continue.';
      if (experimentNameStatus === 'checking') return 'Checking the name...';
      if (experimentNameStatus === 'conflict')
        return `An experiment named ${experimentPreview} already exists.`;
      return errors.newName?.message;
    }
    // The re-run path picks its source on this step too, so the source's own problems are
    // reported here rather than swallowed by a Submit that quietly does nothing.
    if (step === 'evaluation' && mode === MODE_EXPERIMENT) {
      if (hasNoEvaluations) return NO_EVALUATIONS_MESSAGE;
      if (!selectedEvaluation) return 'Pick an evaluation to re-run.';
      if (isValidatingEvaluation) return 'Checking the saved eval config...';
      return evaluationConfigIssue ?? undefined;
    }
    return undefined;
  };
  const blocker = stepBlocker();

  const goNext = () => {
    if (blocker) return;
    setStep(nextStep(steps, step));
  };

  const goBack = () => {
    clearErrors();
    setStep(previousStep(steps, step));
  };

  const onSubmit: SubmitHandler<SubmitEvaluationFormData> = async (formData) => {
    // The resolver has passed; these are the gates held outside form state.
    // isJudgeModelsLoading blocks too: invalidModelRefs is empty while the list is in flight, so
    // submitting inside that window would skip the check entirely.
    if (hasNameConflict) return;
    if (mode === MODE_DEFAULT && (!uploads || (isLlmJudge && isJudgeModelsLoading))) return;
    if (
      mode === MODE_EXPERIMENT &&
      (isValidatingEvaluation || !selectedEvaluation || evaluationConfigIssue)
    )
      return;
    try {
      await submitEvaluation(formData);
    } catch {
      // Error rendered via errorText prop.
    }
  };

  const onLastStep = isLastStep(steps, step);

  // Enter on an intermediate step means "next", not "submit the half-filled form".
  const handleFormSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!onLastStep) {
      goNext();
      return;
    }
    setSubmitAttempted(true);
    void handleSubmit(onSubmit)(event);
  };

  const errorMessage = submitErrorMessage(submitError);

  return (
    <FormModal
      open={open}
      onClose={resetAndClose}
      title="Run Agent Evaluation"
      submitButtonText="Submit"
      onSubmit={handleFormSubmit}
      disabled={isPending}
      loading={isPending}
      className="w-[690px]! max-w-[95vw]!"
      slotFooterRight={
        <Flex gap="2">
          <Button kind="tertiary" type="button" onClick={resetAndClose} disabled={isPending}>
            Cancel
          </Button>
          {stepIndex(steps, step) > 0 && (
            <Button kind="secondary" type="button" onClick={goBack} disabled={isPending}>
              Back
            </Button>
          )}
          {onLastStep ? (
            <LoadingButton color="brand" type="submit" loading={isPending} disabled={isPending}>
              Submit
            </LoadingButton>
          ) : (
            <Button color="brand" type="button" onClick={goNext} disabled={!!blocker || isPending}>
              Next
            </Button>
          )}
        </Flex>
      }
    >
      <FormProvider {...methods}>
        <Stack gap="density-xl">
          <Stepper
            aria-label="Run evaluation progress"
            className="eval-wizard-stepper"
            activeStep={stepIndex(steps, step)}
            items={steps.map((item) => ({ slotHeading: stepHeading(item) }))}
          />

          {step === 'start' && (
            <>
              {agentProp ? (
                <Text kind="body/regular/md">
                  Run evaluation via NeMo Evaluator&apos;s built-in runner. Evaluator also supports
                  Harbor and Gym as runners.{' '}
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
              ) : (
                <ControlledSelect
                  useControllerProps={{ control, name: 'agent' }}
                  loading={isAgentsLoading}
                  items={agents.flatMap((agent) =>
                    agent.name ? [{ value: agent.name, children: agent.name }] : []
                  )}
                  formFieldProps={{
                    slotLabel: 'Agent',
                    slotError: agentFieldError,
                    status: agentFieldError ? 'error' : undefined,
                  }}
                />
              )}

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
              />
            </>
          )}

          {step === 'evaluation' && (
            <>
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

              {isCreateMode && (
                <>
                  <Text className="text-secondary" kind="label/bold/sm">
                    Select evaluation set
                  </Text>
                  <Text className="text-secondary" kind="body/regular/md">
                    Learn more about evaluation set requirements in the{' '}
                    <Anchor
                      kind="inline"
                      textKind="body/regular/md"
                      href={LINK_DOCS_STUDIO_EXPERIMENTS}
                      target="_blank"
                      rel="noreferrer"
                    >
                      evaluation documentation
                    </Anchor>
                    .
                  </Text>

                  <Upload
                    accept=".jsonl,.json"
                    onValueChange={handleDatasetPicked}
                    onFileRemove={clearDatasetPick}
                    status={datasetError ? 'error' : undefined}
                    renderInput={(slotInput) => (
                      <FormField
                        name="dataset"
                        slotLabel="Add Dataset"
                        slotHelp="JSONL, or a JSON array of objects."
                        slotError={datasetError}
                        status={datasetError ? 'error' : undefined}
                      >
                        {datasetPick ? null : slotInput}
                      </FormField>
                    )}
                  />

                  <Upload
                    accept=".json,.yaml,.yml"
                    onValueChange={handleConfigPicked}
                    onFileRemove={clearConfigPick}
                    status={configError ? 'error' : undefined}
                    renderInput={(slotInput) => (
                      <FormField
                        name="evalConfig"
                        slotLabel="Select Evaluator Config"
                        slotHelp="Select a JSON or YAML config."
                        slotError={configError}
                        status={configError ? 'error' : undefined}
                      >
                        {configPick ? null : slotInput}
                      </FormField>
                    )}
                  />

                  {isLlmJudge && (
                    <JudgeModelSelect<SubmitEvaluationFormData>
                      formFieldName="judgeModel"
                      slotLabel={
                        judgeRequired
                          ? 'Override All LLM Models'
                          : 'Override All LLM Models (Optional)'
                      }
                      slotError={invalidModelsError}
                    />
                  )}
                </>
              )}

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
            </>
          )}

          {blocker && step !== 'start' && (
            <Text className="text-secondary" kind="body/regular/sm">
              {blocker}
            </Text>
          )}

          {errorMessage && (
            <Text kind="body/regular/md" className="text-feedback-danger whitespace-normal">
              {errorMessage}
            </Text>
          )}
        </Stack>
      </FormProvider>
    </FormModal>
  );
};
