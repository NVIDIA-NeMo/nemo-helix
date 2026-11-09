// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  KNOWN_MCP_SERVER_STATES,
  type McpServerStatus,
  type McpStatusResponse,
  type McpToolStatus,
} from '@studio/api/agents/types';
// eslint-disable-next-line import/extensions -- plain JSON so the Python contract test can read it too
import fixture from '@studio/mocks/mcpStatusResponse.json';

// The Python side round-trips this fixture through the Pydantic model, so drift on either side fails a test.
const response: McpStatusResponse = fixture;

// `satisfies` ties these key lists to the TS interfaces; the assertions tie them to the fixture.
const SERVER_FIELDS = Object.keys({
  name: true,
  transport: true,
  target: true,
  args: true,
  state: true,
  tools: true,
  error: true,
  stderr: true,
} satisfies Record<keyof McpServerStatus, true>);

const TOOL_FIELDS = Object.keys({
  name: true,
  description: true,
  input_schema: true,
  exposed: true,
} satisfies Record<keyof McpToolStatus, true>);

describe('MCP status contract', () => {
  it('covers every server state the UI knows', () => {
    expect(new Set(response.servers.map((server) => server.state))).toEqual(
      new Set(KNOWN_MCP_SERVER_STATES)
    );
  });

  it('declares exactly the fields the runtime returns', () => {
    expect(Object.keys(response).sort()).toEqual(['checked_at', 'servers']);
    for (const server of response.servers) {
      expect(Object.keys(server).sort()).toEqual([...SERVER_FIELDS].sort());
      for (const tool of server.tools) {
        expect(Object.keys(tool).sort()).toEqual([...TOOL_FIELDS].sort());
      }
    }
  });
});
