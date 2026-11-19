// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Breadcrumbs } from '@nvidia/foundations-react-core';
import { type FC, useMemo } from 'react';

interface BreadcrumbItem {
  /** Display label for the breadcrumb */
  label: string;
  /** Optional click handler for navigation */
  onClick?: () => void;
}

export interface FileBreadcrumbsProps {
  /** Fileset name - becomes the first breadcrumb */
  filesetName: string;
  /** File path - split into breadcrumb segments */
  filePath: string;
  /** Callback when the fileset name (first crumb) is clicked */
  onFilesetClick?: () => void;
  /** Callback when folder is clicked (receives full path to that folder) */
  onFolderClick?: (folderPath: string) => void;
}

/**
 * File breadcrumbs component that automatically generates breadcrumbs from
 * fileset name and file path.
 *
 * Example:
 * - filesetName: "my-dataset"
 * - filePath: "folder1/folder2/file.txt"
 * - Result: "my-dataset > folder1 > folder2 > file.txt"
 *
 * The fileset name and folders (non-last segments) are clickable if callbacks
 * are provided.
 */
export const FileBreadcrumbs: FC<FileBreadcrumbsProps> = ({
  filesetName,
  filePath,
  onFilesetClick,
  onFolderClick,
}) => {
  const breadcrumbs = useMemo((): BreadcrumbItem[] => {
    const items: BreadcrumbItem[] = [
      {
        label: filesetName,
        onClick: onFilesetClick,
      },
    ];

    // Split file path and add each segment
    const pathSegments = filePath.split('/').filter(Boolean);
    pathSegments.forEach((segment, index) => {
      const isLast = index === pathSegments.length - 1;

      // Build the full path up to this segment (for folder navigation)
      const segmentPath = pathSegments.slice(0, index + 1).join('/');

      items.push({
        label: segment,
        // Folders (non-last segments) are clickable if onFolderClick is provided
        onClick: !isLast && onFolderClick ? () => onFolderClick(segmentPath) : undefined,
      });
    });

    return items;
  }, [filesetName, filePath, onFilesetClick, onFolderClick]);

  const items = breadcrumbs.map(({ label, onClick }) => ({
    children:
      onClick !== undefined ? (
        <button
          type="button"
          onClick={onClick}
          className="text-inherit hover:underline cursor-pointer"
        >
          {label}
        </button>
      ) : (
        <span>{label}</span>
      ),
  }));

  return <Breadcrumbs items={items} />;
};
