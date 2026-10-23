// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { GradientBackground } from '@nemo/common/src/components/GradientBackground';
import { Button, Flex, Text, TextArea, Tooltip } from '@nvidia/foundations-react-core';
import { Paperplane, WindowTerminal } from '@nvidia/react-gui-icons/icons';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { writeStoredActiveSessionId } from '@studio/routes/agents/AssistantChatRoute/activeSessionStorage';
import { AssistantLayout } from '@studio/routes/agents/AssistantChatRoute/AssistantLayout';
import type { AssistantChatRouteState } from '@studio/routes/agents/AssistantChatRoute/types';
import { getAssistantChatRoute } from '@studio/routes/utils';
import {
  type ChangeEvent,
  type FC,
  type FormEvent,
  type KeyboardEvent,
  useCallback,
  useState,
} from 'react';
import { useNavigate } from 'react-router';

const LandingComposer = ({
  input,
  onChange,
  onSubmit,
}: {
  input: string;
  onChange: (value: string) => void;
  onSubmit: (prompt: string) => void;
}) => {
  const submitInput = () => {
    const prompt = input.trim();
    if (!prompt) return;

    onSubmit(prompt);
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    submitInput();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || event.shiftKey || event.nativeEvent.isComposing) return;

    event.preventDefault();
    submitInput();
  };

  return (
    <form
      data-testid="dashboard-landing-composer"
      onSubmit={handleSubmit}
      className="w-full rounded-lg border border-base bg-surface-base p-2 shadow-xl"
      data-tour="dashboard-get-started"
    >
      <TextArea
        aria-label="Message NeMo Assistant"
        value={input}
        onChange={(event: ChangeEvent<HTMLTextAreaElement>) => onChange(event.target.value)}
        placeholder="Message NeMo Assistant"
        rows={3}
        resizeable="auto"
        className="max-h-56 w-full border-0 bg-transparent shadow-none focus-within:outline-none focus-within:ring-0 [&:has(:focus-visible)]:outline-none [&:has(:focus-visible)]:ring-0"
        attributes={{
          TextAreaElement: {
            className:
              '[&&]:focus:outline-none [&&]:focus:ring-0 [&&]:focus-visible:outline-none [&&]:focus-visible:ring-0',
            onKeyDown: handleKeyDown,
          },
        }}
      />
      <Flex className="flex items-center justify-between gap-3 px-1 pt-2">
        <Flex className="flex items-center gap-2 text-secondary">
          <WindowTerminal size="16" />
          <Text kind="body/regular/sm">NeMo Assistant</Text>
        </Flex>
        <Tooltip slotContent="Send">
          <Button
            color="brand"
            size="small"
            aria-label="Send message"
            type="submit"
            disabled={!input.trim()}
          >
            <Paperplane size="16" />
          </Button>
        </Tooltip>
      </Flex>
    </form>
  );
};

export const DashboardLandingRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const [input, setInput] = useState('');

  useBreadcrumbs({
    items: [{ slotLabel: 'Dashboard' }],
  });

  const handleSubmit = useCallback(
    (prompt: string) => {
      writeStoredActiveSessionId(workspace, null);
      const state: AssistantChatRouteState = { initialPrompt: prompt };
      navigate(getAssistantChatRoute(workspace), { state });
    },
    [navigate, workspace]
  );

  return (
    <AssistantLayout hideArtifacts>
      <AccessibleTitle title="Dashboard">
        <GradientBackground className="h-full w-full">
          <main className="relative flex h-full w-full items-center justify-center px-4 py-10 text-primary">
            <Flex className="mx-auto flex w-full min-w-0 max-w-4xl flex-col items-center gap-8">
              <Flex className="flex flex-col items-center gap-3 text-center">
                <Text kind="body/bold/2xl" className="text-center">
                  What would you like to do?
                </Text>
              </Flex>

              <LandingComposer input={input} onChange={setInput} onSubmit={handleSubmit} />
            </Flex>
          </main>
        </GradientBackground>
      </AccessibleTitle>
    </AssistantLayout>
  );
};
