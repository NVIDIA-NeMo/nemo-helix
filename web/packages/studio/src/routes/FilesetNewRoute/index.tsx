// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import {
  checkDatasetQuality,
  type DatasetQualityReport,
} from '@nemo/common/src/utils/datasetQuality';
import { handleFormErrorsGeneric } from '@nemo/common/src/utils/forms/error';
import { type FilesetOutput, FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import { Button, SegmentedControl, SidePanel, Stack } from '@nvidia/foundations-react-core';
import { useSampleDatasetFiles } from '@studio/api/datasets/useSampleDatasetFiles';
import { findFreeFilesetName } from '@studio/components/filesets/filesetNames';
import type { NameCheckStatus } from '@studio/components/form/entityNameCheck';
import { SAMPLE_DATASETS, SampleDataset } from '@studio/constants/sampleDatasets';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { DATASET_TYPE_CUSTOM, DATASET_TYPE_SAMPLE } from '@studio/routes/FilesetNewRoute/constants';
import { CustomFilesetForm } from '@studio/routes/FilesetNewRoute/CustomFilesetForm';
import { getSampleDatasetName } from '@studio/routes/FilesetNewRoute/helpers';
import {
  SampleDatasetSection,
  type SampleNameLookup,
} from '@studio/routes/FilesetNewRoute/SampleDatasetSection';
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
  // Once created, the fileset is in the list the name check reads, and must not count against
  // itself while files upload or after an upload fails. A retry after a failed upload reuses it.
  const [createdFileset, setCreatedFileset] = useState<FilesetOutput>();
  // Reported by whichever tab shows the name, so typing does not re-render the whole panel.
  const [nameStatus, setNameStatus] = useState<NameCheckStatus>();
  const nameTaken = nameStatus === 'conflict';
  // The Sample tab has no name field, so its generated name is moved to a free one before it
  // is used. Each lookup is numbered so a slower, older one never overwrites a newer pick.
  const [sampleNameLookup, setSampleNameLookup] = useState<SampleNameLookup>('resolved');
  const sampleNameLookupId = useRef(0);
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
    setError,
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

  /** Names the sample after the first free `{sample}-{workspace}`, `…-2`, `…-3`, so importing
   *  the same sample twice gets a new fileset rather than a name the user cannot change. */
  const applySampleName = useCallback(
    (dataset: SampleDataset) => {
      const lookup = ++sampleNameLookupId.current;
      setValue('name', '', { shouldValidate: false });
      setSampleNameLookup('resolving');
      findFreeFilesetName(workspace, getSampleDatasetName(workspace, dataset.id)).then(
        (name) => {
          if (lookup !== sampleNameLookupId.current) return;
          if (name === null) {
            setSampleNameLookup('exhausted');
            return;
          }
          setValue('name', name, { shouldValidate: false });
          setSampleNameLookup('resolved');
        },
        () => {
          if (lookup === sampleNameLookupId.current) setSampleNameLookup('failed');
        }
      );
    },
    [workspace, setValue]
  );

  // Another create can take the picked name first; with no field to rename it in, move on to
  // the next free name. Once per name, so a lookup that disagrees with the check cannot loop.
  const reResolvedSampleName = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (activeTab !== DATASET_TYPE_SAMPLE || !nameTaken || sampleNameLookup !== 'resolved') return;
    const name = getValues('name');
    if (!name || reResolvedSampleName.current === name) return;
    reResolvedSampleName.current = name;
    applySampleName(selectedSampleDataset);
  }, [activeTab, nameTaken, sampleNameLookup, getValues, applySampleName, selectedSampleDataset]);

  // Sync hidden name/description when a sample is selected (sample tab = simulated local form)
  const handleSelectSample = useCallback(
    (dataset: SampleDataset) => {
      setSelectedSampleDataset(dataset);
      applySampleName(dataset);
      setValue('description', dataset.description ?? '', { shouldValidate: false });
    },
    [applySampleName, setValue]
  );

  // When switching tabs, reset the opposite tab's form state so we don't leak values
  const handleTabChange = useCallback(
    (value: DatasetType) => {
      setActiveTab(value);
      setQualityReports([]);
      if (value === DATASET_TYPE_CUSTOM) {
        sampleNameLookupId.current += 1;
        setSampleNameLookup('resolved');
        setValue('name', '', { shouldValidate: false });
        setValue('description', '', { shouldValidate: false });
        setValue('files', undefined, { shouldValidate: false });
        sampleFilesRef.current = null;
      } else {
        applySampleName(selectedSampleDataset);
        setValue('description', selectedSampleDataset.description ?? '', {
          shouldValidate: false,
        });
        setValue('files', undefined, { shouldValidate: false });
        sampleFilesRef.current = fetchSampleFiles();
      }
    },
    [setValue, applySampleName, selectedSampleDataset, fetchSampleFiles]
  );

  const hasValidationErrors =
    purpose === FilesetPurpose.dataset && qualityReports.some((r) => r.hasErrors);

  const onSubmit = useCreateFileset({
    workspace,
    activeTab,
    storageTab,
    hasValidationErrors,
    nameTaken,
    getValues,
    setError,
    sampleFilesRef,
    setIsSubmitPending,
    createdFileset,
    onCreated: setCreatedFileset,
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
              disabled={
                isSubmitPending ||
                hasValidationErrors ||
                nameTaken ||
                (activeTab === DATASET_TYPE_SAMPLE && sampleNameLookup !== 'resolved')
              }
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
              ownName={createdFileset?.name}
              onNameStatusChange={setNameStatus}
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
              control={control}
              selectedSampleDataset={selectedSampleDataset}
              onSelectSample={handleSelectSample}
              workspace={workspace}
              nameLookup={sampleNameLookup}
              onRetryNameLookup={() => applySampleName(selectedSampleDataset)}
              ownName={createdFileset?.name}
              onNameStatusChange={setNameStatus}
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
