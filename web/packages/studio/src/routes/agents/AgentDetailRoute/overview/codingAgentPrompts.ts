// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

interface CodingAgentPromptParams {
  workspace: string;
  /** The agent this overview is for. Omitted when the agent has not loaded yet. */
  agent?: string;
  /** Origin the coding agent should point the platform CLI and SDK at. */
  baseUrl: string;
}

/** Coding agents `nemo skills install --agent` knows how to write skill files for. */
const SUPPORTED_CODING_AGENTS = 'claude, cursor, codex, opencode';

const AGENT_PLACEHOLDER = '<agent-name>';
const CODING_AGENT_PLACEHOLDER = '<coding-agent>';

const agentName = ({ agent }: CodingAgentPromptParams): string => agent ?? AGENT_PLACEHOLDER;

/**
 * Every prompt opens the same way: what the target platform is, and how to load the skills that
 * own the work. Naming the skill instead of restating its steps keeps these prompts correct as the
 * skills change.
 */
const preamble = (
  params: CodingAgentPromptParams,
  skills: string[],
  opts: { checkExistingSkill?: boolean } = {}
): string[] => {
  const skillList = skills.map((skill) => `\`${skill}\``).join(' and ');
  const many = skills.length > 1;
  const noun = many ? 'skills' : 'skill';
  const them = many ? 'them' : 'it';
  const own = many ? 'own' : 'owns';
  const theyPoint = many ? 'they point' : 'it points';
  const are = many ? 'are' : 'is';

  const installStep = opts.checkExistingSkill
    ? [
        `1. Check whether the ${skillList} ${noun} ${are} already installed for this coding agent (look for its existing skill file, e.g. under \`.claude/skills\`, \`.cursor/rules\`, or the equivalent for this coding agent). If ${them} ${are} already installed, skip straight to step 2 — do not reinstall. Otherwise install ${them} (replace \`${CODING_AGENT_PLACEHOLDER}\` with one of: ${SUPPORTED_CODING_AGENTS}):`,
        `   \`\`\`bash`,
        `   nemo skills install --agent ${CODING_AGENT_PLACEHOLDER}`,
        `   \`\`\``,
        `   If the \`nemo\` CLI is not on PATH here, install it (\`uv tool install nemo-platform\` or \`pip install nemo-platform\`) and re-run.`,
      ]
    : [
        `1. Install the NeMo skills into this coding agent (replace \`${CODING_AGENT_PLACEHOLDER}\` with one of: ${SUPPORTED_CODING_AGENTS}):`,
        `   \`\`\`bash`,
        `   nemo skills install --agent ${CODING_AGENT_PLACEHOLDER}`,
        `   \`\`\``,
        `   If the \`nemo\` CLI is not on PATH here, install it (\`uv tool install nemo-platform\` or \`pip install nemo-platform\`) and re-run.`,
      ];

  return [
    `## Environment`,
    ``,
    `\`\`\`bash`,
    `export NMP_BASE_URL=${params.baseUrl}`,
    `export WORKSPACE=${params.workspace}`,
    `export AGENT_NAME=${agentName(params)}`,
    `\`\`\``,
    ...(params.agent
      ? []
      : [
          ``,
          `Replace \`${AGENT_PLACEHOLDER}\` with the name of the agent in this repository, and use that exact name everywhere below.`,
        ]),
    ``,
    `## How to do it`,
    ``,
    `NeMo Platform ships ${many ? 'skills' : 'a skill'} that ${own} this work end to end. Do not improvise a solution — install the ${noun} and follow ${them} step by step.`,
    ``,
    ...installStep,
    `2. Read the ${skillList} ${noun} and follow the documented steps in order. Run the exact commands documented there and read any \`references/\` file ${theyPoint} at.`,
    `3. Do not invent CLI flags. If a flag you want is not in the skill, check \`nemo <subcommand> --help\`.`,
  ];
};

const closing = [
  ``,
  `## Rules`,
  ``,
  `- Do not report a step complete until you have run its verification and seen it pass. If verification fails or times out, show me what you saw instead of moving on.`,
  `- Make the smallest change that works. Do not refactor the agent while you are here.`,
].join('\n');

/**
 * Prompt for the "Begin with traces" path: get the agent's telemetry into Intake so the overview,
 * insights, and dataset generation have real runs to work from.
 */
export const traceImportPrompt = (params: CodingAgentPromptParams): string =>
  [
    `# Send my agent's telemetry to NeMo Intake`,
    ``,
    `I want to see traces, insights, and generate datasets from real runs of my agent in NeMo Studio.`,
    ``,
    ...preamble(params, ['nemo-intake'], { checkExistingSkill: true }),
    ``,
    `## What I want`,
    ``,
    `- Do not rewrite the agent. Instrument it where it already emits telemetry, or import telemetry it has already produced.`,
    `- If my traces already live in MLflow, LangSmith, Arize Phoenix, or Braintrust, use the skill's importer for that provider instead of adding instrumentation — but still make sure the imported spans end up tagged with the agent name \`${agentName(params)}\` (remap it during import if the source system recorded a different name).`,
    `- Otherwise pick the ingest format that matches what the agent actually emits — OTLP (OpenInference or OTel GenAI semantic conventions), chat completions, or ATIF. Read the skill's comparison table before choosing; do not default to generic OpenTelemetry spans.`,
    `- Every span must carry the agent name \`${agentName(params)}\` exactly as it is named in NeMo Platform (\`gen_ai.agent.name\`, or \`llm.agent.name\` / \`agent.name\` depending on the convention your instrumentation emits). This is the name I created the agent under in Studio, not whatever name the agent's own code, framework, or class defaults to — override it if they differ. Studio's agent overview is keyed on this exact value — if it is missing, inconsistent, or does not match \`${agentName(params)}\` verbatim, the traces land but the agent page stays empty.`,
    `- Set a stable session ID so multi-turn runs group into one session.`,
    `- Assume volume. Ingest them in batches, keep going after a single bad record, and tell me how many landed versus how many were skipped and why.`,
    `- My exports are probably not clean. Expect mixed shapes, one-trace-per-file directories, JSONL, gzipped archives, extra wrapper keys, or missing agent names. Write a small throwaway script to normalize them before ingesting rather than asking me to hand-edit files.`,
    `- If the traces are still sitting in a live system, pull them over its API instead of having me download and upload files.`,
    ``,
    `## Done when`,
    ``,
    `1. The agent has been run at least once against real input with instrumentation enabled.`,
    `2. Querying Intake back returns those spans with the expected hierarchy, inputs, outputs, and statuses. Do not stop at page 1 — page through every result (or read \`pagination.total_results\` from the response) and reconcile it against how many records you sent, how many landed, and how many were skipped and why:`,
    `   \`\`\`bash`,
    `   curl -g "$NMP_BASE_URL/apis/intake/v2/workspaces/$WORKSPACE/spans?filter[agent_name]=$AGENT_NAME&page=1&page_size=100"`,
    `   \`\`\``,
    `3. The returned \`agent_name\` matches \`${agentName(params)}\` exactly.`,
    `4. Tell me what to run to keep producing traces going forward, and what I would have to change to switch ingest formats later.`,
    ...(params.agent
      ? [
          ``,
          `## Then turn the traces into insights`,
          ``,
          `Traces on their own do not tell me what my agent is getting wrong — the Analyst reads them and files what recurs as Insights. Once ingestion verifies:`,
          ``,
          `1. Do not wait for the schedule for the first pass. Run the Analyst once now over what you just imported, following the \`nemo-analyst\` skill — install it the same way as above if it is missing.`,
          `2. Only after that manual run finishes, opt the agent into scheduled analysis so every later batch of traces is picked up too. Enabling it first risks a scheduled run firing while the manual pass is still in flight — the platform only guards against overlapping *scheduled* runs, not a manual CLI run:`,
          `   \`\`\`bash`,
          `   nemo insights analysis enable --agent "$AGENT_NAME" --workspace "$WORKSPACE"`,
          `   nemo insights analysis status --agent "$AGENT_NAME" --workspace "$WORKSPACE"`,
          `   \`\`\``,
          `   This stores the model pair the scheduled runs use, so re-run it if I change models later.`,
          `3. Report the Insights it filed by title and id, and tell me if it filed none and why. Filing nothing is a valid outcome; silently skipping this step is not.`,
        ]
      : []),
    closing,
  ].join('\n');

/**
 * Prompt for the "Integrate your agent" path: register a Platform-managed agent config so the
 * agent can be evaluated, optimized, and deployed from Studio.
 */
export const agentIntegrationPrompt = (params: CodingAgentPromptParams): string =>
  [
    `# Integrate my agent with NeMo Platform`,
    ``,
    `I want to evaluate this agent, run automatic optimization on it, and manage its deployments from NeMo Studio.`,
    ``,
    ...preamble(params, ['nemo-agent-config', 'nemo-build-agent']),
    ``,
    `## What I want`,
    ``,
    `- Write the Platform-managed agent config (\`nemo-agents-spec-v1\`) for this agent — \`nemo-agent-config\` owns that file's shape. Keep it as close to how the agent already runs as the spec allows.`,
    `- If the agent is an existing NAT workflow YAML, tell me whether you are deploying it unchanged or migrating it to \`nemo-agents-spec-v1\`, and why, before you change anything.`,
    `- Show me each \`nemo agents create\` and \`nemo agents deploy\` command and wait for my approval before running it.`,
    `- Register the agent under the name \`${agentName(params)}\` in workspace \`${params.workspace}\` so it lines up with the agent page I am looking at.`,
    `- Do not move fields into \`settings\` to silence a validation error. Fix the named field.`,
    ``,
    `## Done when`,
    ``,
    `1. \`nemo agents create\` validates and registers the config.`,
    `2. \`nemo agents deploy\` reaches \`running\`.`,
    `3. \`nemo agents invoke\` against the deployment returns a sensible answer for one real prompt, and you show me the prompt and the response.`,
    `4. Tell me where the config file lives and what I edit to change the model, the instructions, or the tools.`,
    closing,
  ].join('\n');
