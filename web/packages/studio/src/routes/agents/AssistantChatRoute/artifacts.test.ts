// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  createEmptyAssistantChatArtifacts,
  updateAssistantChatArtifactsFromEvent,
  updateAssistantChatArtifactsFromHistoryItems,
  updateAssistantChatArtifactsFromInputSelection,
  updateAssistantChatArtifactsFromSelections,
} from '@studio/routes/agents/AssistantChatRoute/artifacts';

describe('Assistant chat artifacts', () => {
  it.each([
    {
      kind: 'agent' as const,
      input: {},
      value: { agent: 'calculator-agent' },
      expected: { label: 'Agent', value: 'calculator-agent' },
    },
    {
      kind: 'model' as const,
      input: { output_key: 'selected_model' },
      value: { selected_model: 'nvidia/llama-3.3-nemotron-super-49b-v1' },
      expected: { label: 'Model', value: 'nvidia/llama-3.3-nemotron-super-49b-v1' },
    },
    {
      kind: 'dataset_file' as const,
      input: {},
      value: { dataset_fileset: 'evaluation-data', dataset_path: 'inputs/test.jsonl' },
      expected: { label: 'Dataset', value: 'evaluation-data/inputs/test.jsonl' },
    },
    {
      kind: 'eval_config' as const,
      input: {},
      value: { eval_config_fileset: 'agent-evals', eval_config: 'configs/default.yml' },
      expected: { label: 'Eval config', value: 'agent-evals/configs/default.yml' },
    },
  ])(
    'records a $kind picker submission as a selection artifact',
    ({ expected, input, kind, value }) => {
      const artifacts = updateAssistantChatArtifactsFromInputSelection(
        createEmptyAssistantChatArtifacts(),
        { requestId: 'request-1', kind, input },
        value
      );

      expect(artifacts.selections).toEqual([expected]);
    }
  );

  it('keeps the latest streamed assistant model', () => {
    const initial = createEmptyAssistantChatArtifacts();
    const first = updateAssistantChatArtifactsFromEvent(initial, {
      type: 'assistant',
      message: { model: 'claude-sonnet-4-5', content: [] },
    });
    const updated = updateAssistantChatArtifactsFromEvent(first, {
      type: 'assistant',
      message: { model: 'claude-sonnet-4-6', content: [] },
    });

    expect(updated.model).toBeUndefined();
    expect(updated.model_source).toBeUndefined();
    expect(updated.assistant_model).toBe('claude-sonnet-4-6');
  });

  it('promotes agent and selected model answers while preserving assistant model', () => {
    const withCodingModel = updateAssistantChatArtifactsFromEvent(
      createEmptyAssistantChatArtifacts(),
      {
        type: 'assistant',
        message: { model: 'claude-sonnet-4-6', content: [] },
      }
    );

    const withSelections = updateAssistantChatArtifactsFromSelections(
      withCodingModel,
      [
        { header: 'Agent', question: 'Which agent should be used?' },
        { header: 'Model', question: 'Which inference provider and model should be used?' },
        { header: 'Dataset type', question: 'What kind of dataset do you want to generate?' },
      ],
      {
        'Which agent should be used?': 'beach-finder',
        'Which inference provider and model should be used?':
          'nvidia-build - meta/llama-3.3-70b-instruct',
        'What kind of dataset do you want to generate?': 'Text classification',
      }
    );

    expect(withSelections.agent).toBe('beach-finder');
    expect(withSelections.model).toBe('nvidia-build - meta/llama-3.3-70b-instruct');
    expect(withSelections.model_source).toBe('selection');
    expect(withSelections.assistant_model).toBe('claude-sonnet-4-6');
    expect(withSelections.selections).toEqual([
      { label: 'Agent', value: 'beach-finder' },
      { label: 'Model', value: 'nvidia-build - meta/llama-3.3-70b-instruct' },
      { label: 'Dataset', value: 'Text classification' },
    ]);
  });

  it('does not replace the agent name with an agent deployment action', () => {
    const withAgent = updateAssistantChatArtifactsFromSelections(
      createEmptyAssistantChatArtifacts(),
      [{ header: 'Agent', question: 'Which agent should be used?' }],
      { 'Which agent should be used?': 'beach-finder' }
    );

    const withDeploymentAction = updateAssistantChatArtifactsFromSelections(
      withAgent,
      [
        {
          header: 'Agent',
          question: 'The agent already exists. How should it be redeployed?',
        },
      ],
      {
        'The agent already exists. How should it be redeployed?': 'delete + recreate + redeploy',
      }
    );

    expect(withDeploymentAction.agent).toBe('beach-finder');
    expect(withDeploymentAction.selections[0]).toEqual({
      label: 'Agent',
      value: 'beach-finder',
    });
    expect(withDeploymentAction.selections[1]).toMatchObject({
      value: 'delete + recreate + redeploy',
    });
    expect(withDeploymentAction.selections[1]?.label).not.toBe('Agent');
  });

  it('collects relevant tool artifacts from streamed events', () => {
    const artifacts = updateAssistantChatArtifactsFromEvent(
      { ...createEmptyAssistantChatArtifacts(), workspace: 'default' },
      {
        type: 'assistant',
        message: {
          content: [
            {
              type: 'tool_use',
              name: 'Write',
              input: { file_path: 'agents/beach-finder.yml' },
            },
            {
              type: 'tool_use',
              name: 'mcp__nemo_studio__studio_link',
              input: { destination: 'agents', label: 'Agents' },
            },
            {
              type: 'tool_use',
              name: 'mcp__nemo_studio__job_progress',
              input: {
                job_name: 'agent-eval-1',
                job_type: 'agent_evaluation',
                source: 'evaluator',
              },
            },
          ],
        },
      }
    );

    expect(artifacts.files).toEqual([{ action: 'Wrote', path: 'agents/beach-finder.yml' }]);
    expect(artifacts.links).toEqual([
      { label: 'Agents', destination: 'agents', href: '/workspaces/default/agents' },
    ]);
    expect(artifacts.jobs).toEqual([
      {
        name: 'agent-eval-1',
        job_type: 'agent_evaluation',
        source: 'evaluator',
        href: '/workspaces/default/agents/evaluations/agent-eval-1',
      },
    ]);
    expect(artifacts.tools).toEqual([
      'Write',
      'mcp__nemo_studio__studio_link',
      'mcp__nemo_studio__job_progress',
    ]);
  });

  it('does not double encode encoded file paths in studio link artifacts', () => {
    const artifacts = updateAssistantChatArtifactsFromEvent(
      { ...createEmptyAssistantChatArtifacts(), workspace: 'default' },
      {
        type: 'assistant',
        message: {
          content: [
            {
              type: 'tool_use',
              name: 'mcp__nemo_studio__studio_link',
              input: {
                destination: 'fileset_file',
                name: 'training data',
                file_path_encoded: 'nested%2Fexamples.jsonl',
                label: 'Dataset file',
              },
            },
          ],
        },
      }
    );

    expect(artifacts.links).toEqual([
      {
        label: 'Dataset file',
        destination: 'fileset_file',
        href: '/workspaces/default/filesets/training%20data/detail?tab=files&file=nested%2Fexamples.jsonl',
      },
    ]);
  });

  it('links intake spans through the canonical session route', () => {
    const artifacts = updateAssistantChatArtifactsFromEvent(
      { ...createEmptyAssistantChatArtifacts(), workspace: 'default' },
      {
        type: 'assistant',
        message: {
          content: [
            {
              type: 'tool_use',
              name: 'mcp__nemo_studio__studio_link',
              input: {
                destination: 'intake_span',
                label: 'Span',
                session_id: 'session-agent-run-001',
                trace_id: 'trace-agent-run-001',
                span_id: 'span-root-001',
              },
            },
          ],
        },
      }
    );

    expect(artifacts.links).toEqual([
      {
        label: 'Span',
        destination: 'intake_span',
        href: '/workspaces/default/intake/sessions/session-agent-run-001?traceId=trace-agent-run-001&spanId=span-root-001',
      },
    ]);
  });

  it('promotes draft spec name and model over the assistant model', () => {
    const withCodingModel = updateAssistantChatArtifactsFromEvent(
      createEmptyAssistantChatArtifacts(),
      {
        type: 'assistant',
        message: { model: 'claude-sonnet-4-6', content: [] },
      }
    );

    const withSpecModel = updateAssistantChatArtifactsFromEvent(withCodingModel, {
      type: 'assistant',
      message: {
        content: [
          {
            type: 'text',
            text: [
              'Draft Spec: `cat-identifier`',
              'Name: `cat-identifier`',
              '',
              'Model',
              '`cloud, nvidia/llama-3.3-nemotron-super-49b-v1` - default, good reasoning',
              '',
              'Framework',
              'langgraph-nat',
            ].join('\n'),
          },
        ],
      },
    });
    const afterCodeModelUpdate = updateAssistantChatArtifactsFromEvent(withSpecModel, {
      type: 'assistant',
      message: { model: 'claude-opus-4-6', content: [] },
    });

    expect(afterCodeModelUpdate.agent).toBe('cat-identifier');
    expect(afterCodeModelUpdate.model).toBe('cloud, nvidia/llama-3.3-nemotron-super-49b-v1');
    expect(afterCodeModelUpdate.model_source).toBe('spec');
    expect(afterCodeModelUpdate.assistant_model).toBe('claude-opus-4-6');
  });

  it('derives spec artifacts from loaded transcript items', () => {
    const artifacts = updateAssistantChatArtifactsFromHistoryItems(
      createEmptyAssistantChatArtifacts(),
      [
        { kind: 'user', text: 'draft a cat identifier' },
        {
          kind: 'assistant',
          parts: [
            {
              type: 'text',
              text: [
                'Name: `cat-identifier`',
                '',
                'Model',
                '`cloud, nvidia/llama-3.3-nemotron-super-49b-v1`',
              ].join('\n'),
            },
          ],
        },
      ]
    );

    expect(artifacts.agent).toBe('cat-identifier');
    expect(artifacts.model).toBe('cloud, nvidia/llama-3.3-nemotron-super-49b-v1');
    expect(artifacts.model_source).toBe('spec');
  });
});
