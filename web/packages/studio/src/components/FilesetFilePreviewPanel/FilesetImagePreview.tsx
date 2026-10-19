// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useFilesDownloadFile } from '@nemo/sdk/generated/platform/files';
import { Flex, Spinner, Text } from '@nvidia/foundations-react-core';
import { useEffect, useState, type FC } from 'react';

interface FilesetImagePreviewProps {
  workspace: string;
  filesetName: string;
  filePath: string;
  enabled: boolean;
}

const getFileName = (filePath: string): string => filePath.split('/').at(-1) ?? filePath;

interface ImagePreviewProps {
  imageUrl: string;
  filePath: string;
}

const ImagePreview: FC<ImagePreviewProps> = ({ imageUrl, filePath }) => {
  const [imageLoadError, setImageLoadError] = useState(false);

  const revokeImageUrl = () => URL.revokeObjectURL(imageUrl);

  useEffect(() => {
    return () => URL.revokeObjectURL(imageUrl);
  }, [imageUrl]);

  if (imageLoadError) {
    return (
      <Flex align="center" justify="center" className="h-full">
        <Text className="text-danger-base">Error: Image could not be displayed.</Text>
      </Flex>
    );
  }

  return (
    <Flex className="h-full items-center justify-center overflow-auto rounded-lg border border-base bg-surface-raised p-4">
      <img
        src={imageUrl}
        alt={getFileName(filePath)}
        className="max-h-full max-w-full object-contain"
        onLoad={revokeImageUrl}
        onError={() => {
          revokeImageUrl();
          setImageLoadError(true);
        }}
      />
    </Flex>
  );
};

/** Downloads an image through the authenticated API and displays it in the file viewer. */
export const FilesetImagePreview: FC<FilesetImagePreviewProps> = ({
  workspace,
  filesetName,
  filePath,
  enabled,
}) => {
  const {
    data: imageUrl,
    error,
    isLoading,
  } = useFilesDownloadFile<string>(workspace, filesetName, filePath, {
    query: { enabled, select: URL.createObjectURL },
  });

  if (isLoading) {
    return (
      <Flex align="center" justify="center" className="h-full">
        <Spinner size="medium" aria-label="Loading..." />
      </Flex>
    );
  }

  if (error) {
    return (
      <Flex align="center" justify="center" className="h-full">
        <Text className="text-danger-base">Error: {error.message ?? 'Failed to load image'}</Text>
      </Flex>
    );
  }

  if (!imageUrl) {
    return (
      <Flex align="center" justify="center" className="h-full">
        <Text>No content available</Text>
      </Flex>
    );
  }

  return <ImagePreview key={imageUrl} imageUrl={imageUrl} filePath={filePath} />;
};
