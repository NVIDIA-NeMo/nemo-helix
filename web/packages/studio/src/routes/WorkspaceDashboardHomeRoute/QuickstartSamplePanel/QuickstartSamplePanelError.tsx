// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Panel, StatusMessage } from '@nvidia/foundations-react-core';
import { CircleAlert, Loader2 } from 'lucide-react';
import { useState, type FC } from 'react';

export interface QuickstartSamplePanelErrorProps {
  /** Settles once the retry does, which is what clears the button's busy state. */
  onRetry: () => Promise<unknown>;
}

/**
 * Stands in for `QuickstartSamplePanel` when the sample agent cannot be loaded. Scoped to the
 * panel rather than the page: the stat tiles above it are fetched separately and still work.
 */
export const QuickstartSamplePanelError: FC<QuickstartSamplePanelErrorProps> = ({ onRetry }) => {
  // Local, not the query's `isFetching`: the lookup keeps polling in the background, and the
  // button should only read as busy for a retry the user actually asked for.
  const [retrying, setRetrying] = useState(false);

  const handleRetry = async () => {
    setRetrying(true);
    try {
      await onRetry();
    } finally {
      setRetrying(false);
    }
  };

  return (
    <Panel data-testid="quickstart-sample-panel-error">
      <StatusMessage
        size="small"
        slotMedia={<CircleAlert className="size-10" aria-hidden="true" />}
        slotHeading="Couldn't load the sample agent"
        slotSubheading="It may still be setting up, or NeMo Helix couldn't be reached."
        slotFooter={
          <Button kind="secondary" disabled={retrying} aria-busy={retrying} onClick={handleRetry}>
            Retry
            {retrying && <Loader2 className="size-4 animate-spin" aria-hidden="true" />}
          </Button>
        }
      />
    </Panel>
  );
};
