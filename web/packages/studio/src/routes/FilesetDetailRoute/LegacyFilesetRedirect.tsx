// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useQueryParams } from '@nemo/common/src/hooks/useQueryParams';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { QUERY_PARAMETERS } from '@studio/routes/constants';
import { getFilesetRoute, getWorkspaceFilesetsRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { Navigate, useParams } from 'react-router';

/**
 * Redirects the retired fileset side-panel URLs
 * (`/filesets/:filesetId` and `/filesets/:filesetId/file/:filePathEncoded`)
 * to the dedicated fileset detail page so bookmarks and older assistant links
 * keep working.
 */
export const LegacyFilesetRedirect: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { getQueryParam } = useQueryParams();
  const { [ROUTE_PARAMS.filesetId]: filesetId, [ROUTE_PARAMS.filePathEncoded]: filePath } =
    useParams();

  if (!filesetId) {
    return <Navigate to={getWorkspaceFilesetsRoute(workspace)} replace />;
  }

  const folder = getQueryParam(QUERY_PARAMETERS.filesetFolder) || undefined;
  return (
    <Navigate
      to={getFilesetRoute(workspace, filesetId, { folder, file: filePath || undefined })}
      replace
    />
  );
};
