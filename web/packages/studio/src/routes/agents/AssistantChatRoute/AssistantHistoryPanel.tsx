// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Collapse, Expand } from '@nemo/icons/icons';
import { Button, Flex, Tooltip } from '@nvidia/foundations-react-core';
import { AssistantArtifactsPane } from '@studio/routes/agents/AssistantChatRoute/historyPanel/AssistantArtifactsPane';
import { FloatingPanel } from '@studio/routes/agents/AssistantChatRoute/historyPanel/FloatingPanel';
import { HistoryPanelContents } from '@studio/routes/agents/AssistantChatRoute/historyPanel/HistoryPanelContents';
import { SkillsPanelContents } from '@studio/routes/agents/AssistantChatRoute/historyPanel/SkillsPanelContents';
import type { AssistantHistoryPanelProps } from '@studio/routes/agents/AssistantChatRoute/historyPanel/types';
import { useLocalStorage } from '@studio/util/hooks/useLocalStorage';
import {
  ASSISTANT_HISTORY_OPEN_KEY,
  ASSISTANT_OPEN_FLOATING_PANEL_KEY,
} from '@studio/util/localStorage';
import { type FC } from 'react';

type OpenFloatingPanel = 'history' | 'skills';

export const AssistantHistoryPanel: FC<AssistantHistoryPanelProps> = ({
  hideArtifacts,
  ...props
}) => {
  const [historyOpen, setHistoryOpen] = useLocalStorage(ASSISTANT_HISTORY_OPEN_KEY, 'true');
  const [openFloatingPanel, setOpenFloatingPanel, clearOpenFloatingPanel] =
    useLocalStorage<OpenFloatingPanel>(ASSISTANT_OPEN_FLOATING_PANEL_KEY);
  const isOpen = historyOpen !== 'false';
  const toggleLabel = isOpen ? 'Collapse NeMo Assistant history' : 'Expand NeMo Assistant history';
  const handleFloatingPanelOpenChange = (panel: OpenFloatingPanel, open: boolean) => {
    if (open) {
      setOpenFloatingPanel(panel);
      return;
    }
    clearOpenFloatingPanel();
  };

  if (!isOpen) {
    return (
      <aside className="flex shrink-0 justify-center border-t border-base bg-surface-base p-density-xs dark:bg-surface-raised lg:w-14 lg:border-l lg:border-t-0">
        <Tooltip slotContent={toggleLabel} side="left">
          <Button
            aria-label={toggleLabel}
            kind="tertiary"
            size="small"
            type="button"
            onClick={() => setHistoryOpen('true')}
          >
            <Expand size="18" />
          </Button>
        </Tooltip>
      </aside>
    );
  }

  return (
    <aside className="flex min-h-80 w-full shrink-0 flex-col gap-density-md overflow-y-auto bg-transparent py-density-md pl-density-md pr-density-lg lg:w-[30rem] xl:w-[32rem]">
      {!hideArtifacts && (
        <AssistantArtifactsPane
          artifacts={props.artifacts}
          collapseLabel={toggleLabel}
          onCollapse={() => setHistoryOpen('false')}
        />
      )}
      {hideArtifacts && (
        <Flex justify="end">
          <Tooltip slotContent={toggleLabel} side="left">
            <Button
              aria-label={toggleLabel}
              kind="tertiary"
              size="small"
              type="button"
              onClick={() => setHistoryOpen('false')}
            >
              <Collapse size="18" />
            </Button>
          </Tooltip>
        </Flex>
      )}
      <FloatingPanel
        open={openFloatingPanel === 'history'}
        title="All Chats"
        onOpenChange={(open) => handleFloatingPanelOpenChange('history', open)}
      >
        <HistoryPanelContents {...props} />
      </FloatingPanel>
      <FloatingPanel
        open={openFloatingPanel === 'skills'}
        title="Skills"
        onOpenChange={(open) => handleFloatingPanelOpenChange('skills', open)}
      >
        <SkillsPanelContents />
      </FloatingPanel>
    </aside>
  );
};
