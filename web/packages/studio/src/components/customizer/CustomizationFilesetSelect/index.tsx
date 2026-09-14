// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CodeEditor } from '@nemo/common/src/components/CodeEditor';
import { ContentType } from '@nemo/common/src/components/CodeEditor/constants';
import { useFilePreview } from '@nemo/common/src/components/DatasetFileSelect/hooks/useFilePreview';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { FileList, FileListItem } from '@nemo/common/src/components/FileList';
import { ControlledSwitch } from '@nemo/common/src/components/form/ControlledSwitch';
import {
  getEntityReference,
  getPartsFromReference,
  getURNFromNamedEntityRef,
} from '@nemo/common/src/namedEntity';
import {
  useFilesListFilesets as useListFilesets,
  useFilesRetrieveFileset,
} from '@nemo/sdk/generated/platform/files';
import { FilesetOutput as Fileset } from '@nemo/sdk/generated/platform/schema';
import {
  Anchor,
  Block,
  Button,
  FormField,
  SelectContent,
  SelectItem,
  SelectRoot,
  SelectTrigger,
  SidePanel,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { CustomizationFilesetCreateModal } from '@studio/components/CustomizationFilesetCreateModal';
import { FileValidationPanel } from '@studio/components/customizer/CustomizationFilesetSelect/FileValidationPanel';
import { PatternsTooltipTrigger } from '@studio/components/customizer/CustomizationFilesetSelect/FileValidationPanel/PatternsTooltip';
import { Loading } from '@studio/components/Layouts/Loading';
import { LINK_DOCS_FINE_TUNE_DATASET_FORMAT_REQUIREMENTS } from '@studio/constants/links';
import { useCustomizationDatasetValidation } from '@studio/hooks/useCustomizationDatasetValidation';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { inferJsonContentType, isJsonFile } from '@studio/util/files';
import {
  DATASET_FIELD_BY_BACKEND,
  resolveTrainingType,
  type CustomizationFormFields,
} from '@studio/util/forms/customization';
import { Database, FolderOpen } from 'lucide-react';
import { FC, useEffect, useMemo, useState } from 'react';
import { useController, useFormContext, useWatch } from 'react-hook-form';

const NEW_DATASET_VALUE = '__new_dataset__';

export interface CustomizationFilesetSelectProps {
  disabled?: boolean;
}

export const CustomizationFilesetSelect: FC<CustomizationFilesetSelectProps> = ({ disabled }) => {
  const workspace = useWorkspaceFromPath();
  const [openModal, setOpenModal] = useState<'create' | undefined>();

  const { control, setValue } = useFormContext<CustomizationFormFields>();
  const backend = useWatch({ control, name: 'backend' });

  const automodelTrainingType = useWatch({ control, name: 'automodel.training.training_type' });
  const rlTrainingType = useWatch({ control, name: 'grpo.trainingType' });

  const fieldName = DATASET_FIELD_BY_BACKEND[backend];
  const trainingType = resolveTrainingType(backend, automodelTrainingType, rlTrainingType);

  const {
    field: { value: selectedRef, onChange: setSelectedRef, onBlur },
    fieldState: { error: fieldError },
  } = useController({ control, name: fieldName });

  const {
    data: filesetsResponse,
    isPending,
    isFetching,
    error,
  } = useListFilesets(workspace, {
    page_size: 100,
    sort: 'created_at',
    filter: { purpose: 'dataset' },
  });
  const filesets = useMemo(() => filesetsResponse?.data ?? [], [filesetsResponse?.data]);
  const dropdownDisabled = isPending || isFetching || !!error;

  const selectedParts = selectedRef ? getPartsFromReference(selectedRef) : undefined;
  const filesetURN =
    selectedParts?.workspace && selectedParts?.name
      ? getURNFromNamedEntityRef({ workspace: selectedParts.workspace, name: selectedParts.name })
      : undefined;

  const {
    data: fileset,
    status: fetchFilesetStatus,
    refetch: refetchFileset,
  } = useFilesRetrieveFileset(selectedParts?.workspace ?? '', selectedParts?.name ?? '', {
    query: { enabled: !!(selectedParts?.workspace && selectedParts?.name) },
  });

  const validation = useCustomizationDatasetValidation({
    fileset: filesetURN,
    trainingType,
  });
  const { training, validation: validationFiles } = validation;

  const detectedVariant = validation.schema?.variant;
  useEffect(() => {
    if (backend !== 'unsloth' || !detectedVariant) return;
    setValue('unsloth.dataset.apply_chat_template', detectedVariant === 'sft-chat', {
      shouldValidate: false,
    });
  }, [backend, detectedVariant, setValue]);

  /**
   * Automodel and unsloth take a second fileset reference for validation, pointed at the
   * same fileset as training. Cleared when the fileset holds no validation files, so the
   * backend applies the auto-split this picker already announces. RL takes one reference
   * and finds both files inside it.
   *
   * File discovery is the only thing that can tell us whether the fileset carries
   * validation rows, and it reports none both while it is still running and when it has
   * failed. Writing in those states would assert an absence nobody established, and would
   * wipe a reference the user entered by hand on the field itself.
   */
  const { hasValidation, isPending: isDiscovering, discoveryError } = validation;
  useEffect(() => {
    if (backend === 'rl') return;
    const field =
      backend === 'automodel' ? 'automodel.dataset.validation' : 'unsloth.dataset.validation_path';
    const ref = (selectedRef as string) || undefined;
    if (!ref || isDiscovering || discoveryError) return;

    setValue(field, hasValidation ? ref : undefined, { shouldValidate: false });
  }, [backend, hasValidation, isDiscovering, discoveryError, selectedRef, setValue]);

  const onCreate = (createdFileset: Fileset) => {
    setSelectedRef(getEntityReference(createdFileset));
    setOpenModal(undefined);
  };

  const handleSelectChange = (value: string) => {
    if (value === NEW_DATASET_VALUE) {
      setOpenModal('create');
      return;
    }
    const picked = filesets.find((f) => getEntityReference(f) === value);
    if (picked) {
      setSelectedRef(getEntityReference(picked));
    }
  };

  const selectedDropdownValue = (selectedRef as string) ?? '';

  const {
    previewFile,
    previewContent,
    isLoadingPreview,
    previewError,
    setPreviewFile,
    clearPreview,
  } = useFilePreview();

  const filesetFiles = useMemo<FileListItem[]>(() => {
    if (fetchFilesetStatus !== 'success') return [];
    return [...training, ...validationFiles].map((file) => ({
      path: file.path,
      dataset: fileset ?? undefined,
      rowCount: file.rowCount,
    }));
  }, [training, validationFiles, fileset, fetchFilesetStatus]);

  const hasMissingTrainingFiles =
    !validation.hasTraining && !validation.isPending && selectedDropdownValue;

  return (
    <Stack gap="density-2xl">
      <Text kind="body/bold/lg">
        {trainingType === 'grpo' ? 'Training Prompts' : 'Training Data'}
      </Text>
      {trainingType === 'grpo' ? (
        <Text kind="body/regular/md">
          GRPO datasets use NeMo Gym format. Each record must include{' '}
          <code>responses_create_params</code> and <code>agent_ref</code>; agent-specific fields
          such as <code>vf_env_id</code> are optional. Use the <code>pi-to-gym-conversion</code> CLI
          to convert Prime Intellect hub environments into a verifiers dataset.
        </Text>
      ) : (
        <Text kind="body/regular/md">
          Datasets should be in JSONL format and split into separate, representative training and
          validation sets. For formatting guidelines, refer to the{' '}
          <Anchor
            href={LINK_DOCS_FINE_TUNE_DATASET_FORMAT_REQUIREMENTS}
            target="_blank"
            rel="noopener noreferrer"
          >
            Dataset Format Requirements
          </Anchor>
          .
        </Text>
      )}

      <FormField
        slotLabel={trainingType === 'grpo' ? 'Training Prompts' : 'Dataset'}
        slotError={
          fieldError?.message ??
          (hasMissingTrainingFiles
            ? 'No training files were found in this dataset. Customizer needs at least one training file to start fine-tuning.'
            : undefined)
        }
        status={fieldError || hasMissingTrainingFiles ? 'error' : undefined}
      >
        <SelectRoot
          value={selectedDropdownValue}
          onValueChange={handleSelectChange}
          onOpenChange={(open) => {
            if (!open) onBlur();
          }}
          disabled={disabled || dropdownDisabled}
        >
          <SelectTrigger
            aria-label="dataset-select"
            placeholder="Select a dataset"
            slotStart={
              selectedDropdownValue && selectedDropdownValue !== NEW_DATASET_VALUE ? (
                <Database width={16} height={16} className="text-fg-subdued" />
              ) : undefined
            }
          />
          <SelectContent>
            {filesets.map((f) => {
              const ref = getEntityReference(f);
              return (
                <SelectItem key={ref} value={ref}>
                  {f.name ?? ''}
                </SelectItem>
              );
            })}
            <SelectItem className="border-t border-base" value={NEW_DATASET_VALUE}>
              New Dataset
            </SelectItem>
          </SelectContent>
        </SelectRoot>
      </FormField>

      {backend === 'unsloth' && (
        <ControlledSwitch
          useControllerProps={{ name: 'unsloth.dataset.apply_chat_template', control }}
          formFieldProps={{
            slotLabel: 'Apply chat template',
            labelPosition: 'left',
            slotInfo:
              "Render each row's messages with the tokenizer's chat template. Auto-enabled for chat datasets; turn off for datasets that already have a plain text column.",
          }}
          disabled={disabled}
        />
      )}
      {backend === 'rl' && trainingType === 'dpo' && (
        <Text kind="body/regular/md" color="secondary">
          Dataset must contain <strong>training.jsonl</strong> in a preference format (chosen /
          rejected pairs, BinaryPreference, Tulu3, HelpSteer3, or native Preference).{' '}
          <strong>validation.jsonl</strong> is optional — training data is split automatically when
          it is absent.
        </Text>
      )}

      {fetchFilesetStatus === 'pending' && selectedParts?.name && (
        <Block paddingX="density-md">
          <Loading description="Dataset loading..." />
        </Block>
      )}

      {fetchFilesetStatus === 'error' && (
        <Block paddingX="density-md">
          <ErrorMessage
            header="Loading Error"
            message="This dataset was unable to load. Please try again."
            slotFooter={
              <Button type="button" onClick={() => refetchFileset()}>
                Retry
              </Button>
            }
          />
        </Block>
      )}

      {fetchFilesetStatus === 'success' && filesetFiles.length > 0 && (
        <FileList files={filesetFiles} allowDelete={false} onPreviewFile={setPreviewFile} />
      )}

      <PatternsTooltipTrigger
        label={hasMissingTrainingFiles ? undefined : 'How are files matched within the Dataset?'}
      />
      {fetchFilesetStatus === 'success' && <FileValidationPanel validation={validation} />}

      {openModal === 'create' && (
        <CustomizationFilesetCreateModal
          open
          onClose={() => setOpenModal(undefined)}
          onFilesetCreated={onCreate}
        />
      )}

      {previewFile && (
        <SidePanel
          slotHeading={
            <div className="flex gap-2 items-center">
              <FolderOpen />
              {previewFile.dataset
                ? `${previewFile.dataset.workspace}/${previewFile.dataset.name}/${previewFile.path}`
                : previewFile.path}
            </div>
          }
          side="right"
          open
          onOpenChange={clearPreview}
          onEscapeKeyDown={(e) => {
            e.preventDefault();
            clearPreview();
          }}
          onPointerDownOutside={(e) => {
            e.preventDefault();
            clearPreview();
          }}
          attributes={{
            SidePanelHeading: { className: 'font-normal' },
          }}
          bordered
          modal
          className="max-w-[960px] w-full"
        >
          {previewError ? (
            <div className="flex h-full items-center justify-center text-red-600">
              Error loading file: {previewError.message}
            </div>
          ) : isLoadingPreview ? (
            <div className="flex h-full items-center justify-center">
              <span>Loading content...</span>
            </div>
          ) : (
            <div className="h-full">
              <CodeEditor
                content={previewContent ?? ''}
                contentType={
                  isJsonFile(inferJsonContentType(previewFile.path))
                    ? ContentType.JSON
                    : ContentType.TEXT
                }
                readOnly
              />
            </div>
          )}
        </SidePanel>
      )}
    </Stack>
  );
};
