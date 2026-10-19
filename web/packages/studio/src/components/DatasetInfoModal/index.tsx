// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getEntityReference } from '@nemo/common/src/namedEntity';
import { useFilesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { Tooltip, Grid, Label, Modal } from '@nvidia/foundations-react-core';
import { Loading } from '@studio/components/Layouts/Loading';
import { ValueWithLabel } from '@studio/components/ValueWithLabel';
import { formatDateTime } from '@studio/util/date';
import { getHumanReadableFileSize } from '@studio/util/files';
import { FC } from 'react';

export interface DatasetInfoModalProps {
  open: boolean;
  onClose: () => void;
  maxFiles?: number;
  dataset: FilesetOutput;
}

export const DatasetInfoModal: FC<DatasetInfoModalProps> = ({
  open,
  onClose,
  maxFiles = 10,
  dataset,
}) => {
  const { data: filesResponse, isPending } = useFilesListFilesetFiles(
    dataset.workspace,
    dataset.name
  );
  const files = filesResponse?.data;
  const showLoader = isPending || !dataset;
  return (
    <Modal open={open} onOpenChange={onClose} slotHeading={`Dataset: ${dataset.name}`}>
      {showLoader ? (
        <Loading />
      ) : (
        <Grid cols={1} gap="density-md">
          <ValueWithLabel label="Dataset ID" value={getEntityReference(dataset)} />
          {dataset?.name && <ValueWithLabel label="Dataset Name" value={dataset?.name} />}
          {dataset?.description && (
            <ValueWithLabel label="Dataset description" value={dataset?.description} />
          )}
          <Grid cols={2} gap="density-md">
            {dataset?.created_at && (
              <ValueWithLabel label="Created" value={formatDateTime(dataset.created_at || '')} />
            )}
            {dataset?.updated_at && (
              <ValueWithLabel label="Updated" value={formatDateTime(dataset.updated_at || '')} />
            )}
          </Grid>
          {files && files.length > 0 && (
            <div>
              <Label>Files</Label>
              <ul className="ml-4">
                {files.slice(0, maxFiles).map((file, i) => (
                  <li key={`file-${i}`} className="w-fit">
                    <Tooltip slotContent={`File ref: ${file.file_ref}`} side="top">
                      <span>
                        {' '}
                        {file.path} ({getHumanReadableFileSize(file.size)})
                      </span>
                    </Tooltip>
                  </li>
                ))}
                {files.length > maxFiles && (
                  <li key="files-additional">
                    ...and {files.length - maxFiles} additional file(s) omitted from display
                  </li>
                )}
              </ul>
            </div>
          )}
        </Grid>
      )}
    </Modal>
  );
};
