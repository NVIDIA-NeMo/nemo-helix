// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { CodingAgentPromptModal } from '@studio/routes/agents/AgentDetailRoute/overview/CodingAgentPromptModal';
import {
  agentIntegrationPrompt,
  traceImportPrompt,
} from '@studio/routes/agents/AgentDetailRoute/overview/codingAgentPrompts';
import { GetStartedOption } from '@studio/routes/agents/AgentDetailRoute/overview/GetStartedOption';
import { type FC, useState } from 'react';

interface GetStartedPanelProps {
  workspace: string;
  /** Named in the prompts so the coding agent knows which agent to wire up. */
  agentName?: string;
}

const TRACES_HEADING = 'Begin with traces';
const TRACES_DESCRIPTION =
  'Import observability data to generate insights and power dataset generation from traces, without altering agent code.';
const INTEGRATE_HEADING = 'Integrate your agent';
const INTEGRATE_DESCRIPTION =
  'Integrate your agent to NeMo Helix via Fabric to power evaluations, auto optimization, and manage agent deployments.';

/**
 * What the overview shows before an agent reports anything: the two ways to connect it.
 *
 * Both paths are work the user does in their own repository, so each hands off a prompt for their
 * coding agent rather than trying to do it from Studio.
 */
export const GetStartedPanel: FC<GetStartedPanelProps> = ({ workspace, agentName }) => {
  const [openPrompt, setOpenPrompt] = useState<'traces' | 'integrate' | null>(null);
  const params = { workspace, agent: agentName, baseUrl: PLATFORM_BASE_URL };

  return (
    <Stack gap="4">
      <Text kind="title/md">Get started with agent optimization</Text>
      <Flex gap="density-2xl" align="stretch">
        <GetStartedOption
          heading={TRACES_HEADING}
          description={TRACES_DESCRIPTION}
          actionLabel="Get coding agent prompt for importing traces"
          onGetPrompt={() => setOpenPrompt('traces')}
        />
        <GetStartedOption
          heading={INTEGRATE_HEADING}
          description={INTEGRATE_DESCRIPTION}
          actionLabel="Get coding agent prompt for integrating your agent"
          onGetPrompt={() => setOpenPrompt('integrate')}
        />
      </Flex>
      <CodingAgentPromptModal
        open={openPrompt === 'traces'}
        onClose={() => setOpenPrompt(null)}
        heading={TRACES_HEADING}
        description={TRACES_DESCRIPTION}
        prompt={traceImportPrompt(params)}
      />
      <CodingAgentPromptModal
        open={openPrompt === 'integrate'}
        onClose={() => setOpenPrompt(null)}
        heading={INTEGRATE_HEADING}
        description={INTEGRATE_DESCRIPTION}
        prompt={agentIntegrationPrompt(params)}
      />
    </Stack>
  );
};
