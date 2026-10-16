// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledTextArea } from '@nemo/common/src/components/form/ControlledTextArea';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { RadioCard } from '@nemo/common/src/components/RadioCard';
import { type DatasetQualityReport } from '@nemo/common/src/utils/datasetQuality';
import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import {
  Flex,
  Grid,
  RadioGroupRoot,
  Spinner,
  Stack,
  TabsContent,
  TabsTrigger,
  TabsRoot,
  Text,
  TabsList,
  Upload,
} from '@nvidia/foundations-react-core';
import { DatasetQualityReportView } from '@studio/routes/FilesetNewRoute/components/DatasetQualityReportView';
import { PURPOSE_OPTIONS, DATASET_TYPE_CUSTOM } from '@studio/routes/FilesetNewRoute/constants';
import { toFileList } from '@studio/routes/FilesetNewRoute/helpers';
import { DatasetFormFields, DatasetType } from '@studio/routes/FilesetNewRoute/types';
import { EditSecretModal } from '@studio/routes/SecretsListRoute/EditSecretModal';
import { SecretSearchableSelect } from '@studio/routes/SecretsListRoute/SecretSearchableSelect';
import { FC, FormEventHandler, RefObject, useState } from 'react';
import { Control, Controller, FieldErrors, UseFormSetValue } from 'react-hook-form';

interface CustomFilesetFormProps {
  control: Control<DatasetFormFields>;
  errors: FieldErrors<DatasetFormFields>;
  setValue: UseFormSetValue<DatasetFormFields>;
  isSubmitPending: boolean;
  purpose: FilesetPurpose;
  activeTab: DatasetType;
  workspace: string;
  storageTab: 'local' | 'external';
  setStorageTab: (value: 'local' | 'external') => void;
  selectedSecretName: string | undefined;
  secretKeyLabel: string;
  isValidating: boolean;
  qualityReports: DatasetQualityReport[];
  qualityReportRef: RefObject<HTMLDivElement | null>;
  onFormSubmit: FormEventHandler<HTMLFormElement>;
  onFilesChange: (files: File[]) => void;
  onClearQualityReports: () => void;
  onRequestNewSecret: () => void;
}

export const CustomFilesetForm: FC<CustomFilesetFormProps> = ({
  control,
  errors,
  setValue,
  isSubmitPending,
  purpose,
  activeTab,
  workspace,
  storageTab,
  setStorageTab,
  selectedSecretName,
  secretKeyLabel,
  isValidating,
  qualityReports,
  qualityReportRef,
  onFormSubmit,
  onFilesChange,
  onClearQualityReports,
  onRequestNewSecret,
}) => {
  // Local, unlike creation: the parent owns that only because it writes the new name back
  // into the form, and editing changes nothing this form holds.
  const [editSecretName, setEditSecretName] = useState<string | null>(null);

  return (
    <>
      <Text kind="body/regular/md">
        Filesets organize files by purpose. Pick a purpose to control which metadata fields are
        available, give the fileset a name, and choose where its files live.
      </Text>
      <form onSubmit={onFormSubmit}>
        <Stack gap="density-2xl">
          <Flex direction="col" gap="density-lg">
            <ControlledTextInput
              label="Fileset Name"
              disabled={isSubmitPending}
              useControllerProps={{
                name: 'name',
                control,
              }}
            />

            <ControlledTextArea
              label="Description (optional)"
              disabled={isSubmitPending}
              useControllerProps={{ name: 'description', control }}
            />
          </Flex>

          <Flex direction="col" gap="density-lg" className="w-full min-w-0">
            <Text kind="label/bold/lg">Purpose</Text>
            <Text kind="body/regular/md">
              Purpose determines which metadata fields are available and can&apos;t be changed after
              the fileset is created.
            </Text>
            <Controller
              control={control}
              name="purpose"
              render={({ field }) => (
                <RadioGroupRoot
                  name={field.name}
                  className="w-full"
                  value={field.value}
                  onValueChange={(value) => field.onChange(value as FilesetPurpose)}
                >
                  <Grid gap="density-md" cols={1}>
                    {PURPOSE_OPTIONS.map((option) => (
                      <RadioCard
                        key={option.value}
                        value={option.value}
                        label={option.label}
                        description={option.description}
                      />
                    ))}
                  </Grid>
                </RadioGroupRoot>
              )}
            />
          </Flex>

          <Flex direction="col" gap="density-lg" className="w-full min-w-0">
            <Text kind="label/bold/lg">Source</Text>
            <Text kind="body/regular/md">
              Upload files for local read/write access, or provide a URL and a workspace secret for
              external read-only access.
            </Text>

            <TabsRoot
              className="w-full min-w-0"
              value={storageTab}
              onValueChange={(value) => {
                const next = value as 'local' | 'external';
                setStorageTab(next);
                if (next === 'local') {
                  setValue('url', '', { shouldValidate: false });
                  setValue('secretKey', '', { shouldValidate: false });
                } else {
                  onClearQualityReports();
                }
              }}
            >
              <TabsList>
                <TabsTrigger value="local">Upload</TabsTrigger>
                <TabsTrigger value="external">External</TabsTrigger>
              </TabsList>
              <TabsContent className="w-full min-w-0 p-0 items-stretch" value="local">
                <Stack gap="density-md" className="w-full">
                  <Upload
                    accept="text/csv,text/json,.jsonl,.parquet"
                    multiple
                    onValueChange={(files) => {
                      const list = Array.isArray(files) ? files : files ? [files] : undefined;
                      void onFilesChange(toFileList(list));
                    }}
                  >
                    Supports JSONL, JSON, CSV, and Parquet files up to 50 MB.
                  </Upload>
                  {purpose === FilesetPurpose.dataset && (
                    <Stack gap="density-sm" ref={qualityReportRef}>
                      {isValidating && <Spinner size="small" description="Validating dataset…" />}
                      {qualityReports.map((report) => (
                        <DatasetQualityReportView key={report.fileName} report={report} />
                      ))}
                    </Stack>
                  )}
                </Stack>
              </TabsContent>
              <TabsContent className="w-full min-w-0 p-0 items-stretch" value="external">
                <Stack gap="density-lg" className="w-full min-w-0">
                  <ControlledTextInput
                    label="URL"
                    disabled={isSubmitPending}
                    useControllerProps={{ name: 'url', control }}
                  />
                  <SecretSearchableSelect
                    workspace={workspace}
                    queryEnabled={
                      activeTab === DATASET_TYPE_CUSTOM &&
                      storageTab === 'external' &&
                      Boolean(workspace)
                    }
                    ensureOptionValue={selectedSecretName || undefined}
                    useControllerProps={{ control, name: 'secretKey' }}
                    onRequestNewSecret={onRequestNewSecret}
                    onRequestEditSecret={setEditSecretName}
                    triggerPlaceholder=""
                    formFieldProps={{
                      slotLabel: secretKeyLabel,
                      slotInfo:
                        'Select a secret that stores the credential for this URL, or choose New Secret to create one.',
                      slotError: errors.secretKey?.message,
                    }}
                  />
                </Stack>
              </TabsContent>
            </TabsRoot>
          </Flex>
        </Stack>
      </form>
      {editSecretName ? (
        <EditSecretModal
          workspace={workspace}
          name={editSecretName}
          open
          onClose={() => setEditSecretName(null)}
        />
      ) : null}
    </>
  );
};
