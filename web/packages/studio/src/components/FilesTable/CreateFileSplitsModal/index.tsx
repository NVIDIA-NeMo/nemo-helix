// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { ControlledRadioGroup } from '@nemo/common/src/components/form/ControlledRadioGroup';
import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { handleFormErrorsGeneric } from '@nemo/common/src/utils/forms/error';
import { Banner, Divider, Flex, Label, Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import { useDatasetFileContent } from '@studio/api/datasets/useDatasetFileContent';
import { useSplitDatasetFile } from '@studio/api/datasets/useSplitDatasetFile';
import {
  SELECT_SPLIT_OPTIONS,
  splitDescriptorToList,
} from '@studio/components/FilesTable/CreateFileSplitsModal/constants';
import { FileSplitAdvancedOptions } from '@studio/components/FilesTable/CreateFileSplitsModal/FileSplitAdvancedOptions';
import { FileSplitsSliders } from '@studio/components/FilesTable/CreateFileSplitsModal/FileSplitsSliders';
import {
  CreateFileSplitsFormFields,
  createFileSplitsSchema,
} from '@studio/components/FilesTable/CreateFileSplitsModal/types';
import { MiddleTruncatedPath } from '@studio/components/MiddleTruncatedPath';
import { ValueWithLabel } from '@studio/components/ValueWithLabel';
import { useSelectedDatasetId } from '@studio/hooks/useSelectedDatasetId';
import { tooltipClassName } from '@studio/styles/common';
import { getContentSchema } from '@studio/util/files';
import { getTextWithCount } from '@studio/util/strings';
import { Split } from 'lucide-react';
import { ComponentProps, FC, useMemo } from 'react';
import { FormProvider, useForm, useWatch } from 'react-hook-form';

interface Props extends Pick<ComponentProps<typeof FormModal>, 'open' | 'onClose'> {
  filepath?: string;
  datasetId?: string;
  fileOptions?: string[];
}

/**
 * This modal is used to handle splitting a larger file
 * into smaller files for training/validation/evaluation.
 */
export const CreateFileSplitsModal: FC<Props> = ({
  open,
  onClose,
  filepath,
  datasetId,
  fileOptions,
}) => {
  const toast = useToast();
  const resolvedDatasetId = useSelectedDatasetId({ datasetId });
  const datasetNameSplit = getPartsFromReference(resolvedDatasetId);

  const defaultFilepath =
    filepath ??
    fileOptions?.find((path) => /\.(json|jsonl|parquet)$/i.test(path)) ??
    fileOptions?.[0] ??
    '';

  const formMethods = useForm<CreateFileSplitsFormFields>({
    mode: 'onChange',
    resolver: zodResolver(createFileSplitsSchema),
    defaultValues: {
      filepath: defaultFilepath,
      splitDescriptor: SELECT_SPLIT_OPTIONS[0],
      training: 80,
      testing: 20,
      validation: 0,
      distributionType: 'random',
    },
  });

  const { control, reset, handleSubmit, setValue } = formMethods;
  const filepathForm = useWatch({ control, name: 'filepath' });
  const splitDescriptorForm = useWatch({ control, name: 'splitDescriptor' });
  const resetAndClose = () => {
    reset();
    onClose();
  };

  const {
    data: fileContent,
    isLoading: isLoadingFileContent,
    error: fileContentError,
  } = useDatasetFileContent({
    ...datasetNameSplit,
    path: filepathForm,
    fullContent: true,
  });
  const { total_rows } = useMemo(() => {
    const contentSchema = getContentSchema(fileContent, {
      fileType: filepathForm.split('.').at(-1) ?? '',
    });
    setValue('schemaKeys', Object.keys(contentSchema.schema ?? {}));
    return contentSchema;
  }, [fileContent, filepathForm, setValue]);

  const { mutateAsync: splitFile, isPending } = useSplitDatasetFile({
    onSuccess: () => {
      toast.success(`Successfully created file split and added to ${datasetNameSplit.name}.`);
      resetAndClose();
    },
    onError: (error) => {
      toast.error(error.message);
    },
  });

  const onSubmit = async (data: CreateFileSplitsFormFields) => {
    const splits = data.splitDescriptor.includes('Custom')
      ? [
          `${Math.floor(data.training)}%`,
          `${Math.floor(data.testing)}%`,
          `${Math.floor(data.validation)}%`,
        ]
      : splitDescriptorToList[data.splitDescriptor as keyof typeof splitDescriptorToList];
    if (!fileContent) {
      toast.error('File content not found');
      return;
    }
    await splitFile({
      workspace: datasetNameSplit.workspace,
      datasetName: datasetNameSplit.name,
      filepath: filepathForm,
      fileContent: fileContent,
      splits,
      distributionType: data.distributionType,
      seed: data.seed,
      sortKey: data.sortKey,
    });
  };

  return (
    <FormProvider {...formMethods}>
      <FormModal
        open={open}
        title={
          <Flex gap="density-md" align="center">
            <Split />
            Create Split
          </Flex>
        }
        submitButtonText="Confirm"
        onSubmit={handleSubmit(
          onSubmit,
          handleFormErrorsGeneric({ title: 'Create File Splits Form Errors' })
        )}
        onClose={resetAndClose}
        disabled={isPending}
        submitDisabled={isLoadingFileContent || Boolean(fileContentError)}
        loading={isPending}
      >
        <Stack gap="density-xl">
          <Text className="leading-normal">
            To fine-tune and evaluate a model, you need to split a dataset into three subsets:
            training data, validation data, and test data.
          </Text>
          {filepath || !fileOptions?.length ? (
            <ValueWithLabel
              labelProps={{ className: 'font-bold' }}
              label="Source File"
              value={filepathForm}
            />
          ) : (
            <ControlledSelect
              items={fileOptions.map((path) => ({
                value: path,
                children: <MiddleTruncatedPath>{path}</MiddleTruncatedPath>,
              }))}
              renderValue={(value) => <MiddleTruncatedPath>{value}</MiddleTruncatedPath>}
              formFieldProps={{
                attributes: { Popover: { className: tooltipClassName } },
                slotLabel: <Label className="font-bold">Source File</Label>,
              }}
              useControllerProps={{ control, name: 'filepath' }}
            />
          )}
          <Divider />
          {fileContentError ? (
            <Banner
              kind="inline"
              status="error"
              attributes={{ BannerIcon: { className: 'self-start' } }}
            >
              {fileContentError.message}
            </Banner>
          ) : isLoadingFileContent ? (
            <Flex justify="center" align="center" className="h-full py-[80px]">
              <Spinner description="Loading file content..." />
            </Flex>
          ) : (
            <Stack gap="density-xl" className="pb-4">
              <ControlledSelect
                items={SELECT_SPLIT_OPTIONS}
                formFieldProps={{
                  attributes: {
                    Popover: { className: tooltipClassName },
                    FormFieldLabelGroup: { className: 'justify-between' },
                  },
                  slotInfo: (
                    <Stack gap="2">
                      <span>
                        <span className="font-bold">Training %:</span> Set the percentage of the
                        dataset to be allocated for training the model. This should typically cover
                        the largest portion of the data.
                      </span>
                      <span>
                        <span className="font-bold">Validation %:</span> Set the percentage of the
                        dataset to use for validation purposes, which assists in tuning the model's
                        parameters and avoiding overfitting.
                      </span>
                      <span>
                        <span className="font-bold">Testing %: </span> Set the percentage dedicated
                        to testing, allowing for a final, unbiased evaluation of the model on unseen
                        data.
                      </span>
                      <span>
                        <span className="font-bold">Custom %:</span> Provides the flexibility to
                        allocate the dataset according to your specific preferences.
                      </span>
                      <span>Ensure that the total of all percentages equals 100%.</span>
                    </Stack>
                  ),
                  slotLabel: <Label className="font-bold">Split Percentage</Label>,
                }}
                useControllerProps={{ control, name: 'splitDescriptor' }}
              />

              {total_rows && total_rows < 50 && (
                <Banner
                  kind="inline"
                  status="warning"
                  attributes={{ BannerIcon: { className: 'self-start' } }}
                >
                  The source file contains only {getTextWithCount('example', total_rows)}. At least
                  50 samples are recommended for fine-tuning.
                </Banner>
              )}
              {splitDescriptorForm?.includes('Custom') && <FileSplitsSliders />}
              <ControlledRadioGroup
                orientation="horizontal"
                defaultValue="random"
                className="flex w-full! *:flex-1"
                items={[
                  { children: 'Random', value: 'random' },
                  { children: 'Sequential', value: 'sequential' },
                ]}
                useControllerProps={{ name: 'distributionType', control }}
                formFieldProps={{
                  attributes: {
                    Popover: { className: tooltipClassName },
                    FormFieldLabelGroup: { className: 'justify-between' },
                    FormFieldContentGroup: { className: 'gap-2.5' },
                  },
                  slotLabel: <Label className="font-bold">Distribution Type</Label>,
                  slotInfo: (
                    <Stack gap="2">
                      <span>
                        <span className="font-bold">Random Splits:</span> Use this when the sequence
                        does not impact model performance.
                      </span>
                      <span>
                        <span className="font-bold">Sequential Splits:</span> Use this for language
                        tasks involving sequence dependencies, such as predictive text applications,
                        dialogue generation, or any scenario relying on historical text data to
                        anticipate future interactions.
                      </span>
                    </Stack>
                  ),
                }}
              />
              <FileSplitAdvancedOptions distributionType={formMethods.watch('distributionType')} />
            </Stack>
          )}
        </Stack>
      </FormModal>
    </FormProvider>
  );
};
