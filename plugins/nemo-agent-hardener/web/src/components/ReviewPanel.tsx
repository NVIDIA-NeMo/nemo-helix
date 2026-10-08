// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BenignSuiteEditor } from '@agent-hardener/components/BenignSuiteEditor';
import type { SuiteRow } from '@agent-hardener/components/hitlTypes';
import { Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { FC, useState } from 'react';

interface ReviewPanelProps {
  suite: SuiteRow[];
  loading?: boolean;
  onSubmit: (suite: SuiteRow[]) => void;
}

// The benign-suite review step, inline (as a run tab). Reuses the full BenignSuiteEditor (edit + add/remove)
// so the operator can shape the suite before approving it for the war-game to replay.
export const ReviewPanel: FC<ReviewPanelProps> = ({ suite, loading, onSubmit }) => {
  const [rows, setRows] = useState<SuiteRow[]>(suite);

  return (
    <div className="flex h-full flex-col">
      <Stack gap="density-xs" className="mb-4 shrink-0">
        <Text kind="body/semibold/lg">Review the benign suite</Text>
        <Text kind="body/regular/md" className="text-subtle">
          Edit or drop the generated requests. The approved suite is replayed against the agent to
          confirm it still works after hardening.
        </Text>
      </Stack>

      <div className="min-h-0 flex-1 overflow-auto pr-density-xs">
        <BenignSuiteEditor value={rows} onChange={setRows} disabled={loading} />
      </div>

      <Flex className="mt-4 shrink-0 justify-end">
        <Button color="brand" disabled={loading} onClick={() => onSubmit(rows)}>
          {loading ? 'Approving…' : `Approve ${rows.length} request${rows.length === 1 ? '' : 's'}`}
        </Button>
      </Flex>
    </div>
  );
};
