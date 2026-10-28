// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface ChatErrorDescription {
  title: string;
  hint: string;
}

// The agent chat path flattens every failure into one string (the platform returns 502 for any
// agent failure and the model's HTTP status survives only inside the message). These rules read
// that text back into something a user can act on. Envelope phrases are written by the platform
// (Fabric server, agents gateway) and the `Error code: NNN` is the openai SDK inside the Fabric
// runtime. Keep rules matching on those, never on provider-authored prose, which can change.
//
// ponytail: text matching until the platform returns a structured error code; replace these
// rules with a switch on that code then.

// Optional leading `\d{3} ` is the openai client prefixing the HTTP status, so match anywhere.
const UPSTREAM_STATUS = /Error code: (\d{3})\b/;
// The provider's message is Python-repr'd inside the error, so quotes may arrive backslash-escaped.
const RETIRED_MODEL = /The model \\?'([^']+?)\\?' has reached its end of life/;
const NO_BODY = /^(\d{3}) status code \(no body\)$/;

const describeUpstreamStatus = (
  status: number,
  message: string
): ChatErrorDescription | undefined => {
  switch (true) {
    case status === 410: {
      const model = RETIRED_MODEL.exec(message)?.[1];
      return {
        title: model
          ? `The model "${model}" has been retired.`
          : "The agent's model has been retired.",
        hint: 'Update the agent to use an available model, then redeploy it.',
      };
    }
    case status === 404:
      return {
        title: "The agent's model was not found.",
        hint: 'Check the model name in the agent config and that its provider is registered.',
      };
    case status === 401 || status === 403:
      return {
        title: "The model provider rejected the agent's credentials.",
        hint: 'Check the provider API key secret the agent uses.',
      };
    case status === 424:
      return {
        title: 'The model provider rejected the request.',
        hint: 'Check the model is still offered by its provider and that the provider is ready.',
      };
    case status === 429:
      return {
        title: 'The model provider is rate limiting requests.',
        hint: 'Wait a moment and try again.',
      };
    case status >= 500:
      return {
        title: 'The model provider is unavailable.',
        hint: 'This is usually temporary. Try again shortly.',
      };
    default:
      return undefined;
  }
};

/**
 * Turns a chat failure message into an actionable title and hint, or `undefined` when no rule
 * matches (the caller then keeps the raw message and the generic banner).
 */
export const describeChatError = (message: string): ChatErrorDescription | undefined => {
  const upstreamStatus = UPSTREAM_STATUS.exec(message)?.[1];
  if (upstreamStatus) return describeUpstreamStatus(Number(upstreamStatus), message);

  if (message.includes('Fabric runtime startup failed')) {
    return {
      title: "The agent's runtime failed to start.",
      hint: 'Check the deployment logs for the adapter or MCP server error.',
    };
  }

  if (message.includes('Could not connect to agent')) {
    return {
      title: 'The agent did not respond.',
      hint: "It may have stopped or run past the gateway's response timeout. Check the deployment status and logs.",
    };
  }

  const noBodyStatus = NO_BODY.exec(message)?.[1];
  if (noBodyStatus) {
    return {
      title: `The platform returned ${noBodyStatus} with no details.`,
      hint: 'The response did not come from the agent. Check that the platform is reachable and no proxy or network layer is blocking the request.',
    };
  }

  return undefined;
};
