// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { Banner, Divider, Flex, Stack } from '@nvidia/foundations-react-core';
import { ScmBranch } from '@nvidia/react-gui-icons/icons';
import { useDatasetFileContent } from '@studio/api/datasets/useDatasetFileContent';
import {
  isTransformableFilePath,
  useDatasetFileTransform,
} from '@studio/api/datasets/useDatasetFileTransform';
import { DiscardTransformModal } from '@studio/components/transform/DiscardTransformModal';
import { FormatPicker } from '@studio/components/transform/FormatPicker';
import { MappingSection } from '@studio/components/transform/MappingSection';
import { TransformPreview } from '@studio/components/transform/TransformPreview';
import { useTransformMapping } from '@studio/components/transform/useTransformMapping';
import { ValueWithLabel } from '@studio/components/ValueWithLabel';
import { useSelectedDatasetId } from '@studio/hooks/useSelectedDatasetId';
import { getContentColumns } from '@studio/util/files';
import { useMemo, useState, type ComponentProps, type FC } from 'react';

interface Props extends Pick<ComponentProps<typeof FormModal>, 'open' | 'onClose'> {
  filepath?: string;
  datasetId?: string;
}

/**
 * Rewrites one fileset file into another schema, in place. Same field mapping as
 * the Data Designer transform, but applied in the browser: the file is small
 * enough to read, remap, and re-upload without a job.
 */
export const TransformFileModal: FC<Props> = ({ open, onClose, filepath, datasetId }) => {
  const toast = useToast();
  const resolvedDatasetId = useSelectedDatasetId({ datasetId });
  const datasetNameSplit = getPartsFromReference(resolvedDatasetId);
  const [isDiscardOpen, setIsDiscardOpen] = useState(false);

  const resolvedFilepath = filepath ?? '';
  const filepathParts = resolvedFilepath.split('.');
  const fileType = filepathParts.length > 1 ? (filepathParts.at(-1) ?? '') : '';
  const isSupportedFile = isTransformableFilePath(resolvedFilepath);

  const { data: fileContent, isLoading: isLoadingFileContent } = useDatasetFileContent({
    ...datasetNameSplit,
    path: resolvedFilepath,
  });

  const columns = useMemo(() => getContentColumns(fileContent, fileType), [fileContent, fileType]);
  const mapping = useTransformMapping({ columns });

  const { mutate: transformFile, isPending } = useDatasetFileTransform({
    onSuccess: () => {
      toast.success('Successfully finished file transformation!');
      onClose();
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!fileContent) {
      toast.error('File content not found');
      return;
    }
    transformFile({
      workspace: datasetNameSplit.workspace,
      datasetName: datasetNameSplit.name,
      filepath: resolvedFilepath,
      template: mapping.template,
      fileContent,
      generatedIdColumn: mapping.needsGeneratedId ? mapping.generatedIdColumn : undefined,
    });
  };

  const handleCloseRequest = () => {
    if (mapping.isDirty) {
      setIsDiscardOpen(true);
      return;
    }
    onClose();
  };

  return (
    <>
      <FormModal
        open={open}
        title={
          <Flex gap="density-md" align="center">
            <ScmBranch />
            Transform
          </Flex>
        }
        instruction="Rewrite this file into another schema. Every row is remapped in place — the file is overwritten with the result."
        submitButtonText="Transform file"
        onSubmit={handleSubmit}
        className="w-[860px] overflow-hidden"
        onClose={handleCloseRequest}
        disabled={isPending}
        submitDisabled={isLoadingFileContent || !mapping.isComplete || !isSupportedFile}
        loading={isPending}
      >
        <Stack gap="density-xl">
          {!isSupportedFile && (
            <Banner kind="inline" status="warning">
              This file cannot be transformed: the transform rewrites rows as JSONL, so only .jsonl
              files can be transformed in place.
            </Banner>
          )}

          <ValueWithLabel
            labelProps={{ className: 'font-bold' }}
            label="Source file"
            value={filepath}
          />

          <FormatPicker mapping={mapping} />

          <Divider />

          <MappingSection mapping={mapping} isLoadingColumns={isLoadingFileContent} />

          <TransformPreview
            fileContent={fileContent}
            fileType={fileType}
            template={mapping.template}
            generatedIdColumn={mapping.needsGeneratedId ? mapping.generatedIdColumn : undefined}
          />
        </Stack>
      </FormModal>

      {isDiscardOpen && (
        <DiscardTransformModal
          onClose={() => setIsDiscardOpen(false)}
          onConfirm={onClose}
          description="Your field mapping has not been submitted. Closing now discards it — the file is left unchanged."
        />
      )}
    </>
  );
};
