// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  deleteSession,
  useGetSessionDeletionPermission,
} from '@nemo/sdk/generated/platform/sessions';
import { Button, Stack, Text } from '@nvidia/foundations-react-core';
import { invalidateSessionDeletion } from '@studio/api/intake/invalidateSessionDeletion';
import { getIntakeTracesRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import { useState } from 'react';
import { useNavigate } from 'react-router';

interface SessionDeleteButtonProps {
  workspace: string;
  sessionId: string;
  traceCount?: number;
  spanCount?: number;
}

export const SessionDeleteButton = ({
  workspace,
  sessionId,
  traceCount,
  spanCount,
}: SessionDeleteButtonProps) => {
  const [open, setOpen] = useState(false);
  const [failure, setFailure] = useState<string>();
  const toast = useToast();
  const client = useQueryClient();
  const navigate = useNavigate();
  const { data: permission } = useGetSessionDeletionPermission(workspace, sessionId, {
    query: { retry: false },
  });

  if (!permission?.allowed) return null;

  const finish = async (missing: boolean) => {
    toast.success(
      missing ? 'Session is missing or already removed.' : 'Session telemetry deleted.'
    );
    navigate(getIntakeTracesRoute(workspace), { replace: true });
    await invalidateSessionDeletion(client, workspace);
  };

  return (
    <Stack gap="density-sm">
      <Button kind="secondary" color="danger" onClick={() => setOpen(true)}>
        {failure ? 'Retry session telemetry deletion' : 'Delete session telemetry'}
      </Button>
      {failure && <Text role="alert">{failure}</Text>}
      <DeleteConfirmationModal
        open={open}
        onClose={() => setOpen(false)}
        title="Delete session telemetry?"
        entity="session telemetry"
        ids={[sessionId]}
        description={`Workspace: ${workspace}. Session: ${sessionId}. Delete ALL current traces and spans in this session, including sibling traces of the selected trace. Traces: ${traceCount ?? 'unavailable'}. Spans: ${spanCount ?? 'unavailable'}. Annotations and evaluator results are retained. Later ingestion can recreate telemetry. This cannot be undone.`}
        suppressResultToasts
        onDelete={async () => {
          setFailure(undefined);
          try {
            await deleteSession(workspace, sessionId);
            await finish(false);
            return true;
          } catch (error) {
            const status = isAxiosError(error) ? error.response?.status : undefined;
            if (status === 404) {
              await finish(true);
              return true;
            }
            const guidance =
              status === 503
                ? 'Cleanup may be incomplete. Retry with this same workspace and session, even if the session can no longer be loaded.'
                : status === 403
                  ? 'Deletion requires intake.sessions.delete, workspace access, and write scope (workspace Admin by default).'
                  : status === 409
                    ? 'Deletion is blocked by an evaluation association, conflicting ownership, or competing cleanup. Resolve the reported conflict before retrying.'
                    : 'Deletion failed. You can retry with this same workspace and session.';
            setFailure(`${guidance} ${error instanceof Error ? getErrorMessage(error) : ''}`);
            return false;
          }
        }}
      />
    </Stack>
  );
};
