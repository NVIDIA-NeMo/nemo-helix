// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import {
  checkDatasetQuality,
  type DatasetQualityReport,
} from '@nemo/common/src/utils/datasetQuality';
import { handleFormErrorsGeneric } from '@nemo/common/src/utils/forms/error';
import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import { Button, SegmentedControl, SidePanel, Stack } from '@nvidia/foundations-react-core';
import { useSampleDatasetFiles } from '@studio/api/datasets/useSampleDatasetFiles';
import { SAMPLE_DATASETS, SampleDataset } from '@studio/constants/sampleDatasets';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { DATASET_TYPE_CUSTOM, DATASET_TYPE_SAMPLE } from '@studio/routes/FilesetNewRoute/constants';
import { CustomFilesetForm } from '@studio/routes/FilesetNewRoute/CustomFilesetForm';
import { getSampleDatasetName } from '@studio/routes/FilesetNewRoute/helpers';
import { SampleDatasetSection } from '@studio/routes/FilesetNewRoute/SampleDatasetSection';
import {
  DatasetCreateFilesetFormSchema,
  DatasetFormFields,
  DatasetType,
} from '@studio/routes/FilesetNewRoute/types';
import { useCreateFileset } from '@studio/routes/FilesetNewRoute/useCreateFileset';
import { CreateSecretModal } from '@studio/routes/SecretsListRoute/CreateSecretModal';
import { getWorkspaceFilesetsRoute } from '@studio/routes/utils';
import { isHuggingFaceUrl, isNgcUrl } from '@studio/util/storageConfigFromUrl';
import { QueryObserverResult } from '@tanstack/react-query';
import { FC, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useForm } from 'react-hook-form';
import { useNavigate } from 'react-router';

export const FilesetNewRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const [activeTab, setActiveTab] = useState<DatasetType>(DATASET_TYPE_CUSTOM);
  const [selectedSampleDataset, setSelectedSampleDataset] = useState<SampleDataset>(
    SAMPLE_DATASETS[0]
  );
  const [isSubmitPending, setIsSubmitPending] = useState(false);
  const [qualityReports, setQualityReports] = useState<DatasetQualityReport[]>([]);
  const [isValidating, setIsValidating] = useState(false);
  const qualityReportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (qualityReports.length > 0) {
      qualityReportRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [qualityReports]);

  const navigate = useNavigate();
  const sampleFilesRef = useRef<Promise<QueryObserverResult<File[], Error>> | null>(null);

  const [storageTab, setStorageTab] = useState<'local' | 'external'>('local');
  const [createSecretModalOpen, setCreateSecretModalOpen] = useState(false);

  const {
    control,
    handleSubmit,
    setValue,
    getValues,
    watch,
    formState: { errors },
  } = useForm<DatasetFormFields>({
    resolver: zodResolver(DatasetCreateFilesetFormSchema),
    defaultValues: {
      name: '',
      description: '',
      files: undefined,
      url: '',
      secretKey: '',
      // Default chosen for least surprise: the page previously hardcoded purpose=dataset.
      purpose: FilesetPurpose.dataset,
    },
    mode: 'onChange',
  });

  const url = watch('url');
  const purpose = watch('purpose');

  useEffect(() => {
    if (purpose !== FilesetPurpose.dataset) {
      setQualityReports([]);
      setIsValidating(false);
    }
  }, [purpose]);

  const selectedSecretName = watch('secretKey');
  const secretKeyLabel = useMemo(() => {
    if (!url?.trim()) return 'Secret Key';
    try {
      if (isHuggingFaceUrl(new URL(url))) return 'HuggingFace Token (optional)';
      if (isNgcUrl(new URL(url))) return 'NGC API Key';
    } catch {
      // invalid URL
    }
    return 'Secret Key';
  }, [url]);

  const { refetch: fetchSampleFiles } = useSampleDatasetFiles({
    sampleDataset: selectedSampleDataset,
    enabled: false,
  });

  /**
   * Runs dataset quality checks on newly selected JSONL files and updates the report state.
   * Only runs when purpose is 'dataset'; clears reports for other purposes or non-JSONL files.
   */
  const handleFilesChange = useCallback(
    async (files: File[]) => {
      setValue('files', files, { shouldValidate: false });

      if (purpose !== FilesetPurpose.dataset) {
        setQualityReports([]);
        return;
      }

      const jsonlFiles = files.filter((f) => f.name.endsWith('.jsonl'));
      if (jsonlFiles.length === 0) {
        setQualityReports([]);
        return;
      }

      setIsValidating(true);
      const reports = await Promise.all(jsonlFiles.map(checkDatasetQuality));
      setQualityReports(reports);
      setIsValidating(false);
    },
    [purpose, setValue]
  );

  // Sync hidden name/description when a sample is selected (sample tab = simulated local form)
  const handleSelectSample = useCallback(
    (dataset: SampleDataset) => {
      setSelectedSampleDataset(dataset);
      setValue('name', getSampleDatasetName(workspace, dataset.id), { shouldValidate: false });
      setValue('description', dataset.description ?? '', { shouldValidate: false });
    },
    [workspace, setValue]
  );

  // When switching tabs, reset the opposite tab's form state so we don't leak values
  const handleTabChange = useCallback(
    (value: DatasetType) => {
      setActiveTab(value);
      setQualityReports([]);
      if (value === DATASET_TYPE_CUSTOM) {
        setValue('name', '', { shouldValidate: false });
        setValue('description', '', { shouldValidate: false });
        setValue('files', undefined, { shouldValidate: false });
        sampleFilesRef.current = null;
      } else {
        setValue('name', getSampleDatasetName(workspace, selectedSampleDataset.id), {
          shouldValidate: false,
        });
        setValue('description', selectedSampleDataset.description ?? '', {
          shouldValidate: false,
        });
        setValue('files', undefined, { shouldValidate: false });
        sampleFilesRef.current = fetchSampleFiles();
      }
    },
    [
      setValue,
      workspace,
      selectedSampleDataset.id,
      selectedSampleDataset.description,
      fetchSampleFiles,
    ]
  );

  const hasValidationErrors =
    purpose === FilesetPurpose.dataset && qualityReports.some((r) => r.hasErrors);

  const onSubmit = useCreateFileset({
    workspace,
    activeTab,
    storageTab,
    hasValidationErrors,
    getValues,
    sampleFilesRef,
    setIsSubmitPending,
  });

  const handleClose = useCallback(() => {
    navigate(getWorkspaceFilesetsRoute(workspace), { flushSync: true });
  }, [navigate, workspace]);

  const handleSecretCreated = useCallback(
    (secretName: string) => {
      setValue('secretKey', secretName, { shouldValidate: true });
    },
    [setValue]
  );

  return (
    <>
      <SidePanel
        slotHeading="Create Fileset"
        side="right"
        open
        onOpenChange={handleClose}
        className="max-w-[600px] w-full"
        bordered
        modal
        slotFooter={
          <>
            <Button kind="tertiary" onClick={handleClose}>
              Cancel
            </Button>
            <Button
              type="button"
              color="brand"
              disabled={isSubmitPending || hasValidationErrors}
              onClick={handleSubmit(
                onSubmit,
                handleFormErrorsGeneric({ title: 'Fileset New Form Errors' })
              )}
            >
              {isSubmitPending ? 'Creating Fileset...' : 'Create Fileset'}
            </Button>
          </>
        }
      >
        <Stack className="h-full overflow-auto w-full" gap="density-xl">
          <SegmentedControl
            size="tiny"
            className="w-full"
            value={activeTab}
            onValueChange={(value) => handleTabChange(value as DatasetType)}
            items={[
              { value: DATASET_TYPE_CUSTOM, children: 'Custom Fileset' },
              { value: DATASET_TYPE_SAMPLE, children: 'Sample Dataset' },
            ]}
          />
          {activeTab === DATASET_TYPE_CUSTOM && (
            <CustomFilesetForm
              control={control}
              errors={errors}
              setValue={setValue}
              isSubmitPending={isSubmitPending}
              purpose={purpose}
              activeTab={activeTab}
              workspace={workspace}
              storageTab={storageTab}
              setStorageTab={setStorageTab}
              selectedSecretName={selectedSecretName}
              secretKeyLabel={secretKeyLabel}
              isValidating={isValidating}
              qualityReports={qualityReports}
              qualityReportRef={qualityReportRef}
              onFormSubmit={handleSubmit(
                onSubmit,
                handleFormErrorsGeneric({ title: 'Fileset New Form Errors' })
              )}
              onFilesChange={handleFilesChange}
              onClearQualityReports={() => setQualityReports([])}
              onRequestNewSecret={() => setCreateSecretModalOpen(true)}
            />
          )}
          {activeTab === DATASET_TYPE_SAMPLE && (
            <SampleDatasetSection
              selectedSampleDataset={selectedSampleDataset}
              onSelectSample={handleSelectSample}
            />
          )}
        </Stack>
      </SidePanel>
      <CreateSecretModal
        workspace={workspace}
        open={createSecretModalOpen}
        onClose={() => setCreateSecretModalOpen(false)}
        onSecretCreated={handleSecretCreated}
      />
    </>
  );
};
