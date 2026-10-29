// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface ChatErrorDescription {
  title: string;
  hint: string;
}

/** Turns a chat failure message into an actionable description, or `undefined` if it is unknown. */
export type DescribeChatError = (message: string) => ChatErrorDescription | undefined;

export const GENERIC_CHAT_ERROR = 'There was an error generating a response.';

// Agent chat flattens every failure into one string: the platform returns 502 for any agent
// failure and the model's HTTP status survives only inside the message. These rules read that
// text back into something a user can act on. They match phrases our own code writes (the Fabric
// server and the agents gateway) and the envelope the Fabric runtime wraps an openai SDK failure
// in (`RuntimeError: internal error: APIStatusError: Error code: NNN - ...`), never
// provider-authored prose, which can change.
//
// ponytail: text matching until the platform returns a structured error code; replace these
// rules with a switch on that code then.

// The openai client may prefix the HTTP status (`502 ...`); the platform phrases are anchored at
// the start so a message that merely quotes one of them somewhere is not misread.
const RUNTIME_START_FAILED = /^(?:\d{3} )?Fabric runtime startup failed/;
const AGENT_UNREACHABLE = /^(?:\d{3} )?Could not connect to agent/;
const NO_BODY = /^(\d{3}) status code \(no body\)$/;
// Anchored to the whole envelope so a status that only appears inside quoted detail is ignored.
const UPSTREAM_STATUS =
  /^(?:\d{3} )?RuntimeError: internal error: APIStatusError: Error code: (\d{3}) - /;
// The provider's message is Python-repr'd inside the error, so quotes may arrive backslash-escaped.
const RETIRED_MODEL = /The model \\?'([^']+?)\\?' has reached its end of life/;

const describeUpstreamStatus = (
  status: number,
  message: string
): ChatErrorDescription | undefined => {
  if (status === 410) {
    const model = RETIRED_MODEL.exec(message)?.[1];
    return {
      title: model
        ? `The model "${model}" has been retired.`
        : "The agent's model has been retired.",
      hint: 'Update the agent to use an available model, then redeploy it.',
    };
  }
  if (status === 404) {
    return {
      title: "The agent's model was not found.",
      hint: 'Check the model name in the agent config and that its provider is registered.',
    };
  }
  if (status === 401 || status === 403) {
    return {
      title: "The model provider rejected the agent's credentials.",
      hint: 'Check the provider API key secret the agent uses.',
    };
  }
  if (status === 424) {
    return {
      title: 'The model provider rejected the request.',
      hint: 'Check the model is still offered by its provider and that the provider is ready.',
    };
  }
  if (status === 429) {
    return {
      title: 'The model provider is rate limiting requests.',
      hint: 'Wait a moment and try again.',
    };
  }
  if (status >= 500) {
    return {
      title: 'The model provider is unavailable.',
      hint: 'This is usually temporary. Try again shortly.',
    };
  }
  return undefined;
};

/**
 * Describes an agent chat failure. Platform phrases are checked before the upstream status so a
 * startup or connection failure that happens to quote a status code is not blamed on the model
 * provider. Statuses without a rule (400, 409, 422...) return `undefined`.
 */
export const describeChatError: DescribeChatError = (message) => {
  if (RUNTIME_START_FAILED.test(message)) {
    return {
      title: "The agent's runtime failed to start.",
      hint: 'Check the deployment logs for the adapter or MCP server error.',
    };
  }
  if (AGENT_UNREACHABLE.test(message)) {
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
  const upstreamStatus = UPSTREAM_STATUS.exec(message)?.[1];
  if (upstreamStatus) return describeUpstreamStatus(Number(upstreamStatus), message);
  return undefined;
};

/**
 * The message-body copy for a failed chat reply: the description's title and hint, or the generic
 * line when there is no `describe` or it does not recognise the message. The raw error stays in
 * the message status for the error banner.
 */
export const humanizeChatError = (message: string, describe?: DescribeChatError): string => {
  const described = describe?.(message);
  return described ? `${described.title} ${described.hint}` : GENERIC_CHAT_ERROR;
};
