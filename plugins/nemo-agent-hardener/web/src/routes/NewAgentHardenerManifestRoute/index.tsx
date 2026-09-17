// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { useAgentsForSelect } from '@agent-hardener/api/agents';
import { useInspectAgent } from '@agent-hardener/api/filesets';
import { ModelGroupFields } from '@agent-hardener/components/ModelGroupFields';
import { ProjectManifestWizard } from '@agent-hardener/components/ProjectManifestWizard';
import { parseEnvPairs, splitList } from '@agent-hardener/formValues';
import {
  getAgentHardenerListManifestsQueryKey,
  useAgentHardenerCreateManifest,
  useAgentHardenerGetModelConfigDefaults,
} from '@agent-hardener/generated/api';
import type { WarGameModels } from '@agent-hardener/generated/schema';
import { useBreadcrumbs, useToast, useWorkspace } from '@agent-hardener/host';
import { getAgentHardenerManifestListRoute, getAgentHardenerRunListRoute } from '@agent-hardener/paths';
import {
  manifestFormSchema,
  NAME_PATTERN,
  type ManifestFormData,
} from '@agent-hardener/routes/NewAgentHardenerManifestRoute/schema';
import { AccessibleTitle, AccordionSection, ControlledSelect, ControlledTextInput } from '@nemo/common';
import {
  AccordionRoot,
  Button,
  Flex,
  Panel,
  PageHeader,
  SegmentedControl,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { useQueryClient } from '@tanstack/react-query';
import { FC, useEffect, useMemo, useState } from 'react';
import { useForm } from 'react-hook-form';
import { Link, useNavigate } from 'react-router';

type Source = 'agent' | 'project';

export const NewAgentHardenerManifestRoute: FC = () => {
  const workspace = useWorkspace();
  const navigate = useNavigate();
  const toast = useToast();
  const queryClient = useQueryClient();
  const [source, setSource] = useState<Source>('agent');
  const [models, setModels] = useState<WarGameModels>({});
  const { data: modelDefaults } = useAgentHardenerGetModelConfigDefaults(workspace, { query: {} });

  useBreadcrumbs({
    items: [
      { href: getAgentHardenerRunListRoute(workspace), slotLabel: 'Agent Hardener' },
      { href: getAgentHardenerManifestListRoute(workspace), slotLabel: 'Manifests' },
      { slotLabel: 'New' },
    ],
  });

  const { control, handleSubmit, watch, setError, setValue } = useForm<ManifestFormData>({
    defaultValues: { name: '', agent: '', egress: '', env: '', port: '', secrets: '' },
    resolver: zodResolver(manifestFormSchema),
  });
  const nameValue = watch('name').trim();
  const nameValid = NAME_PATTERN.test(nameValue);

  // Pre-fill the port + secret fields from the agent's auto-derived defaults when one is selected;
  // both stay editable so the operator can override.
  const selectedAgent = watch('agent');
  const inspectAgent = useInspectAgent();
  const { mutate: runInspectAgent } = inspectAgent;
  useEffect(() => {
    if (source !== 'agent' || !selectedAgent) return;
    runInspectAgent(
      { workspace, agent: selectedAgent },
      {
        onSuccess: (facts) => {
          setValue('port', String(facts.port));
          setValue('secrets', facts.secrets.join(', '));
        },
      }
    );
  }, [source, selectedAgent, workspace, runInspectAgent, setValue]);

  const { data: agents = [], isLoading: agentsLoading } = useAgentsForSelect(workspace);
  const agentItems = useMemo(
    () =>
      agents.flatMap((agent) => (agent.name ? [{ value: agent.name, children: agent.name }] : [])),
    [agents]
  );

  const createManifest = useAgentHardenerCreateManifest({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: getAgentHardenerListManifestsQueryKey(workspace) });
        toast.success('Manifest created.');
        navigate(getAgentHardenerManifestListRoute(workspace));
      },
      onError: () => toast.error('Failed to create the manifest. Check the agent and try again.'),
    },
  });

  const onSubmitAgent = handleSubmit((data) => {
    if (!data.agent) {
      setError('agent', { message: 'Select a deployed agent' });
      return;
    }
    const egress = splitList(data.egress);
    const secrets = splitList(data.secrets);
    const env = parseEnvPairs(data.env);
    const port = data.port ? Number(data.port) : undefined;
    if (port !== undefined && !Number.isInteger(port)) {
      setError('port', { message: 'Enter a whole number' });
      return;
    }
    createManifest.mutate({
      workspace,
      data: {
        name: data.name,
        source_type: 'agent',
        agent: data.agent,
        ...(egress.length ? { egress } : {}),
        ...(secrets.length ? { secrets } : {}),
        ...(Object.keys(env).length ? { env } : {}),
        ...(port !== undefined ? { port } : {}),
        models,
      },
    });
  });

  return (
    <AccessibleTitle title="New Agent Hardener manifest">
      <Stack className="h-full overflow-auto" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading="New Manifest"
          slotDescription="Scaffold a reusable war-game target. Give it an ID, then pick where the agent comes from."
        />
        <Panel className="max-w-[720px]">
          <Stack gap="density-xl" padding="density-lg">
            <ControlledTextInput
              useControllerProps={{ control, name: 'name' }}
              formFieldProps={{
                slotLabel: 'Manifest ID',
                slotHelp: 'Lowercase, e.g. clockbot-hardening.',
              }}
            />

            <Stack gap="density-sm">
              <Text kind="body/semibold/sm">Agent Source</Text>
              <SegmentedControl
                className="w-full"
                value={source}
                onValueChange={(v) => setSource(v as Source)}
                items={[
                  { value: 'agent', children: 'Deployed Agent' },
                  { value: 'project', children: 'Upload Project' },
                ]}
              />
            </Stack>

            {source === 'agent' ? (
              <form onSubmit={onSubmitAgent}>
                <Stack gap="density-xl">
                  <ControlledSelect
                    useControllerProps={{ control, name: 'agent' }}
                    loading={agentsLoading}
                    items={agentItems}
                    formFieldProps={{ slotLabel: 'Deployed Agent' }}
                  />
                  <ControlledTextInput
                    useControllerProps={{ control, name: 'egress' }}
                    formFieldProps={{
                      slotLabel: 'Egress Allow-list (optional)',
                      slotHelp:
                        'Comma-separated host[:port] for external services the agent calls (e.g. ' +
                        'en.wikipedia.org, raw.githubusercontent.com). Needed when the tool hosts are not ' +
                        'discoverable from the workflow config.',
                    }}
                  />
                  <ControlledTextInput
                    useControllerProps={{ control, name: 'port' }}
                    formFieldProps={{
                      slotLabel: 'Victim Port',
                      slotHelp: inspectAgent.isPending
                        ? 'Detecting from the agent…'
                        : 'Auto-detected from the deployment. Edit to override.',
                    }}
                  />
                  <ControlledTextInput
                    useControllerProps={{ control, name: 'secrets' }}
                    formFieldProps={{
                      slotLabel: 'Secret Names',
                      slotHelp: inspectAgent.isPending
                        ? 'Detecting from the agent…'
                        : 'Comma-separated; auto-detected from the agent config. Edit to override.',
                    }}
                  />
                  <ControlledTextInput
                    useControllerProps={{ control, name: 'env' }}
                    formFieldProps={{
                      slotLabel: 'Environment Variables (optional)',
                      slotHelp:
                        'Comma-separated KEY=VALUE for non-secret settings the agent reads. ' +
                        'Credentials belong in Secret Names — values here are stored in plain text.',
                    }}
                  />
                  <AccordionRoot>
                    <AccordionSection value="models" title="Models (optional)">
                      <Stack gap="density-md">
                        <Text kind="body/regular/sm" className="text-subtle">
                          Defaults shown as placeholders; override any group for this target. Each
                          run can still change them.
                        </Text>
                        <ModelGroupFields
                          value={models}
                          onChange={setModels}
                          workspace={workspace}
                          defaults={modelDefaults}
                        />
                      </Stack>
                    </AccordionSection>
                  </AccordionRoot>
                  <Flex gap="density-md">
                    <Button color="brand" type="submit" disabled={createManifest.isPending}>
                      {createManifest.isPending ? 'Creating…' : 'Create Manifest'}
                    </Button>
                    <Button asChild kind="tertiary">
                      <Link to={getAgentHardenerManifestListRoute(workspace)}>Cancel</Link>
                    </Button>
                  </Flex>
                </Stack>
              </form>
            ) : (
              <ProjectManifestWizard
                workspace={workspace}
                manifestName={nameValue}
                nameValid={nameValid}
                isCreating={createManifest.isPending}
                onCreate={(values) =>
                  createManifest.mutate({
                    workspace,
                    data: { name: nameValue, source_type: 'project', ...values },
                  })
                }
              />
            )}
          </Stack>
        </Panel>
      </Stack>
    </AccessibleTitle>
  );
};
