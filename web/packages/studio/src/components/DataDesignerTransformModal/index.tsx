// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { useDataDesignerCreateJob } from '@nemo/sdk/generated/data-designer/data-designer';
import {
  Banner,
  Divider,
  Flex,
  Label,
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectRoot,
  SelectTrigger,
  Stack,
  Text,
  TextInput,
} from '@nvidia/foundations-react-core';
import { useDatasetFileContent } from '@studio/api/datasets/useDatasetFileContent';
import { buildTransformJobRequest } from '@studio/components/DataDesignerTransformModal/buildTransformJobRequest';
import { DiscardTransformModal } from '@studio/components/transform/DiscardTransformModal';
import { slugify } from '@studio/components/transform/draft';
import { FormatPicker } from '@studio/components/transform/FormatPicker';
import { OUTPUT_FORMATS } from '@studio/components/transform/formats';
import { MappingSection } from '@studio/components/transform/MappingSection';
import { TransformPreview } from '@studio/components/transform/TransformPreview';
import { useTransformMapping } from '@studio/components/transform/useTransformMapping';
import { getDataDesignerJobDetailsRoute } from '@studio/routes/utils';
import { getContentColumns, getFileNameFromPath } from '@studio/util/files';
import { GitBranch } from 'lucide-react';
import { useCallback, useMemo, useState, type FC } from 'react';
import { useNavigate } from 'react-router';

export interface DataDesignerTransformModalProps {
  open: boolean;
  onClose: () => void;
  /** Workspace the new job is created in. */
  workspace: string;
  /** Name of the job whose output is being transformed, used to seed the new job's name. */
  sourceJobName: string;
  /** Location of the source job's artifacts fileset. */
  filesetWorkspace: string;
  filesetName: string;
  /** Data files in that fileset the transform can read. */
  fileOptions: readonly string[];
  /** Row count of the source job — the transform reads at most this many rows. */
  defaultNumRecords: number;
}

const pickDefaultFile = (fileOptions: readonly string[]): string =>
  fileOptions.find((path) => /\.parquet$/i.test(path)) ?? fileOptions[0] ?? '';

/**
 * Rewrites a finished Data Designer dataset into another schema by launching a
 * second, generation-free Data Designer job: the source file becomes the seed,
 * no columns are declared, and a `schema_transform` processor does the mapping.
 * Because nothing is generated, the job needs no model and costs no inference.
 */
export const DataDesignerTransformModal: FC<DataDesignerTransformModalProps> = ({
  open,
  onClose,
  workspace,
  sourceJobName,
  filesetWorkspace,
  filesetName,
  fileOptions,
  defaultNumRecords,
}) => {
  const toast = useToast();
  const navigate = useNavigate();

  const [filePath, setFilePath] = useState(() => pickDefaultFile(fileOptions));
  const [processorName, setProcessorName] = useState(OUTPUT_FORMATS[0].defaultProcessorName);
  const [jobName, setJobName] = useState(
    () => `${slugify(sourceJobName)}-${OUTPUT_FORMATS[0].defaultProcessorName}`
  );
  const [numRecords, setNumRecords] = useState(String(defaultNumRecords));
  const [isDiscardOpen, setIsDiscardOpen] = useState(false);

  // Parquet is decoded to JSONL by the hook, so every file parses as JSONL here.
  const { data: fileContent, isLoading: isLoadingContent } = useDatasetFileContent({
    workspace: filesetWorkspace,
    name: filesetName,
    path: filePath,
    enabled: open && Boolean(filesetWorkspace && filesetName && filePath),
  });

  const fileType = /\.parquet$/i.test(filePath) ? 'jsonl' : (filePath.split('.').at(-1) ?? '');
  const columns = useMemo(() => getContentColumns(fileContent, fileType), [fileContent, fileType]);

  const handleFormatChange = useCallback(
    (format: (typeof OUTPUT_FORMATS)[number]) => {
      setProcessorName(format.defaultProcessorName);
      setJobName(`${slugify(sourceJobName)}-${format.defaultProcessorName}`);
    },
    [sourceJobName]
  );
  const mapping = useTransformMapping({ columns, onFormatChange: handleFormatChange });

  const parsedRows = Number(numRecords);
  const isRowCountValid = Number.isInteger(parsedRows) && parsedRows > 0;
  const exceedsSource = isRowCountValid && parsedRows > defaultNumRecords;

  const isDirty =
    mapping.isDirty ||
    jobName !== `${slugify(sourceJobName)}-${mapping.format.defaultProcessorName}` ||
    processorName !== mapping.format.defaultProcessorName ||
    numRecords !== String(defaultNumRecords);

  const createJob = useDataDesignerCreateJob();
  const submitError = createJob.error ? getErrorMessage(createJob.error) : null;

  const canSubmit =
    Boolean(filePath) &&
    Boolean(jobName.trim()) &&
    Boolean(processorName.trim()) &&
    isRowCountValid &&
    mapping.isComplete;

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }
    try {
      const created = await createJob.mutateAsync({
        workspace,
        data: buildTransformJobRequest({
          jobName: jobName.trim(),
          processorName: processorName.trim(),
          filesetWorkspace,
          filesetName,
          filePath,
          numRecords: parsedRows,
          template: mapping.template,
          generatedIdColumn: mapping.needsGeneratedId ? mapping.generatedIdColumn : undefined,
        }),
      });
      toast.success(`Transform job ${created?.name ?? jobName} created.`);
      onClose();
      if (created?.name) {
        navigate(getDataDesignerJobDetailsRoute(workspace, created.name));
      }
    } catch {
      // Surfaced through `submitError` in the modal footer.
    }
  };

  /**
   * Every dismissal — Cancel, Escape, the close affordance, a click outside —
   * arrives here through `FormModal`'s single `onClose`, so the field mapping
   * cannot be lost to a stray keypress.
   */
  const handleCloseRequest = () => {
    if (isDirty) {
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
            <GitBranch />
            Transform
          </Flex>
        }
        instruction="Rewrite this dataset into another schema. This starts a second Data Designer job that only maps fields — no rows are generated and no model is called."
        submitButtonText="Create transform job"
        errorText={submitError}
        submitDisabled={!canSubmit}
        loading={createJob.isPending}
        disabled={createJob.isPending}
        onSubmit={handleSubmit}
        onClose={handleCloseRequest}
        className="w-[860px]"
      >
        <Stack gap="density-xl">
          <Stack gap="density-sm">
            <Label className="font-bold">Source file</Label>
            <SelectRoot value={filePath} onValueChange={setFilePath}>
              <SelectTrigger
                placeholder="Select a file"
                aria-label="Source file"
                renderValue={(value) =>
                  typeof value === 'string' ? getFileNameFromPath(value) : null
                }
              />
              <SelectContent className="w-(--radix-popper-anchor-width)">
                <SelectListbox>
                  {fileOptions.map((path) => (
                    <SelectItem key={path} value={path}>
                      {path}
                    </SelectItem>
                  ))}
                </SelectListbox>
              </SelectContent>
            </SelectRoot>
          </Stack>

          <FormatPicker mapping={mapping} />

          <Divider />

          <MappingSection mapping={mapping} isLoadingColumns={isLoadingContent} />

          <Stack gap="density-sm">
            <TransformPreview
              fileContent={fileContent}
              fileType={fileType}
              template={mapping.template}
              generatedIdColumn={mapping.needsGeneratedId ? mapping.generatedIdColumn : undefined}
            />
            <Text kind="body/regular/xs" className="text-muted">
              Written to <code>processors-files/{processorName || 'output'}/</code> in the new
              job&apos;s fileset.
            </Text>
            {mapping.needsGeneratedId && (
              <Text kind="body/regular/xs" className="text-muted">
                <code>{mapping.generatedIdColumn}</code> is not in the source file — the job adds it
                as a UUID sampler column, one value per row. Samplers run without a model, so this
                still costs no inference.
              </Text>
            )}
          </Stack>

          <Divider />

          <Flex gap="density-lg" align="start" className="flex-wrap">
            <Stack gap="density-sm" className="min-w-[240px] flex-1">
              <Label className="font-bold">Job name</Label>
              <TextInput
                value={jobName}
                aria-label="Job name"
                onChange={(event) => setJobName(event.currentTarget.value)}
              />
            </Stack>
            <Stack gap="density-sm" className="min-w-[200px] flex-1">
              <Label className="font-bold">Output name</Label>
              <TextInput
                className="font-mono"
                value={processorName}
                aria-label="Output name"
                onChange={(event) => setProcessorName(event.currentTarget.value)}
              />
            </Stack>
            <Stack gap="density-sm" className="w-[160px]">
              <Label className="font-bold">Rows</Label>
              <TextInput
                type="number"
                value={numRecords}
                aria-label="Rows to transform"
                attributes={{ Input: { min: 1, step: 1 } }}
                onChange={(event) => setNumRecords(event.currentTarget.value)}
              />
            </Stack>
          </Flex>

          {exceedsSource && (
            <Banner kind="inline" status="warning">
              More rows than the source has: the source job produced {defaultNumRecords} rows.
              Reading more than that restarts at the top of the file, duplicating rows in the
              output.
            </Banner>
          )}
        </Stack>
      </FormModal>

      {isDiscardOpen && (
        <DiscardTransformModal
          onClose={() => setIsDiscardOpen(false)}
          onConfirm={onClose}
          description="Your field mapping has not been submitted. Closing now discards it — no job is created."
        />
      )}
    </>
  );
};
