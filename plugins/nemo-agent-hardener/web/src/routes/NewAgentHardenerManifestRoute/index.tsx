// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAgentsForSelect } from '@agent-hardener/api/agents';
import {
  useInspectAgent,
  useInspectProject,
  useUploadProjectFileset,
  type InspectProjectResult,
} from '@agent-hardener/api/filesets';
import { ModelGroupFields } from '@agent-hardener/components/ModelGroupFields';
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
  type ManifestFormData,
  type ManifestSource,
} from '@agent-hardener/routes/NewAgentHardenerManifestRoute/schema';
import { SelectionGuard } from '@agent-hardener/routes/NewAgentHardenerManifestRoute/selectionGuard';
import { zodResolver } from '@hookform/resolvers/zod';
import {
  AccessibleTitle,
  AccordionSection,
  ControlledSelect,
  ControlledTextInput,
  FileUpload,
  RadioCard,
} from '@nemo/common';
import {
  AccordionRoot,
  Button,
  Flex,
  Panel,
  PageHeader,
  RadioGroupRoot,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { useQueryClient } from '@tanstack/react-query';
import { FC, useEffect, useMemo, useRef, useState } from 'react';
import { useForm } from 'react-hook-form';
import { Link, useNavigate } from 'react-router';

/** The harnesses agent-hardener knows how to stage; `other` is the honest escape hatch. */
const HARNESS_ITEMS = ['deepagents', 'hermes', 'langchain', 'langgraph', 'other'].map((value) => ({
  value,
  children: value,
}));

export const NewAgentHardenerManifestRoute: FC = () => {
  const workspace = useWorkspace();
  const navigate = useNavigate();
  const toast = useToast();
  const queryClient = useQueryClient();
  const [models, setModels] = useState<WarGameModels>({});
  const [source, setSource] = useState<ManifestSource>('agent');
  const [projectFile, setProjectFile] = useState<File | null>(null);
  const [projectFileset, setProjectFileset] = useState('');
  const [derived, setDerived] = useState<InspectProjectResult | null>(null);
  const [relayConfirmed, setRelayConfirmed] = useState(false);
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

  // Guards against a late agent/project inspection overwriting a newer selection's form state.
  const selectionGuard = useRef(new SelectionGuard()).current;

  // Pre-fill the port + secret fields from the agent's auto-derived defaults when one is selected;
  // both stay editable so the operator can override.
  const selectedAgent = watch('agent');
  const inspectAgent = useInspectAgent();
  const { mutate: runInspectAgent } = inspectAgent;
  useEffect(() => {
    if (!selectedAgent) return;
    const generation = selectionGuard.begin();
    runInspectAgent(
      { workspace, agent: selectedAgent },
      {
        onSuccess: (facts) => {
          if (!selectionGuard.isCurrent(generation)) return;
          setValue('port', String(facts.port));
          setValue('secrets', facts.secrets.join(', '));
          setValue('egress', facts.egress.join(', '));
        },
      }
    );
  }, [selectedAgent, workspace, runInspectAgent, setValue, selectionGuard]);

  const uploadProject = useUploadProjectFileset();
  const inspectProject = useInspectProject();

  /**
   * Upload the bundle, then read it back. Pre-fills every field the project states so the form only
   * asks for the rest — the difference between bringing an image and authoring a manifest.
   */
  const onProjectSelected = (file: File) => {
    setProjectFile(file);
    setDerived(null);
    const generation = selectionGuard.begin();
    uploadProject.mutate(
      { workspace, manifestName: watch('name') || 'byo', file },
      {
        onSuccess: (ref) => {
          if (!selectionGuard.isCurrent(generation)) return;
          setProjectFileset(ref);
          inspectProject.mutate(
            { workspace, projectFileset: ref },
            {
              onSuccess: (facts) => {
                if (!selectionGuard.isCurrent(generation)) return;
                setDerived(facts);
                setValue('dockerfile', facts.dockerfile);
                setValue('startCommand', facts.start_command);
                setValue('binaries', facts.binaries.join(', '));
                setValue('port', String(facts.port));
                setValue('secrets', facts.secrets.join(', '));
                setValue('egress', facts.egress.join(', '));
                setValue(
                  'env',
                  Object.entries(facts.env)
                    .map(([key, value]) => `${key}=${value}`)
                    .join(', ')
                );
              },
              onError: () => toast.error('Could not read the project bundle.'),
            }
          );
        },
        onError: () => toast.error('Could not upload the project bundle.'),
      }
    );
  };

  /** A field the project could not state, so the form has to ask for it. */
  const asks = (field: string) => derived?.unresolved.includes(field) ?? false;

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

  const onSubmit = handleSubmit((data) => {
    if (source === 'agent' && !data.agent) {
      setError('agent', { message: 'Select a deployed agent' });
      return;
    }
    if (source === 'project' && !projectFileset) {
      toast.error('Upload the project bundle first.');
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
    const binaries = splitList(data.binaries);
    createManifest.mutate({
      workspace,
      data: {
        name: data.name,
        ...(source === 'agent'
          ? { agent: data.agent }
          : {
              source_type: 'project',
              project_fileset: projectFileset,
              relay_integration_confirmed: relayConfirmed,
              ...(data.dockerfile ? { dockerfile: data.dockerfile } : {}),
              ...(data.startCommand ? { start_command: data.startCommand } : {}),
              ...(binaries.length ? { binaries } : {}),
              ...(data.harness ? { harness: data.harness } : {}),
            }),
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

            <form onSubmit={onSubmit}>
                <Stack gap="density-xl">
                  <RadioGroupRoot
                    name="manifest-source"
                    value={source}
                    onValueChange={(value: string) => setSource(value as ManifestSource)}
                  >
                    <Flex gap="density-md" data-testid="manifest-source">
                      <RadioCard
                        value="agent"
                        checked={source === 'agent'}
                        label="Registered agent"
                        description="An agent already registered on the platform."
                      />
                      <RadioCard
                        value="project"
                        checked={source === 'project'}
                        label="Bring your own"
                        description="An image whose Dockerfile you own. We read it and ask only for the rest."
                      />
                    </Flex>
                  </RadioGroupRoot>

                  {source === 'agent' ? (
                    <ControlledSelect
                      useControllerProps={{ control, name: 'agent' }}
                      loading={agentsLoading}
                      items={agentItems}
                      formFieldProps={{ slotLabel: 'Deployed Agent' }}
                    />
                  ) : (
                    <Stack gap="density-lg" data-testid="project-upload">
                      <FileUpload
                        accept={{ 'application/zip': ['.zip'] }}
                        files={projectFile ? [projectFile] : []}
                        onDropAccepted={(files: File[]) => files[0] && onProjectSelected(files[0])}
                        onRemoveFile={() => {
                          // Invalidates any upload/inspect still in flight for the removed file, so
                          // its onSuccess cannot restore what the user just cleared.
                          selectionGuard.invalidate();
                          setProjectFile(null);
                          setProjectFileset('');
                          setDerived(null);
                        }}
                        helperText={
                          uploadProject.isPending
                            ? 'Uploading…'
                            : inspectProject.isPending
                              ? 'Reading the Dockerfile…'
                              : 'A .zip of the directory holding your Dockerfile.'
                        }
                      />
                      {derived?.warnings.map((warning) => (
                        <Text key={warning} kind="body/regular/sm" className="text-subtle">
                          ! {warning}
                        </Text>
                      ))}
                      {derived && (
                        <ControlledTextInput
                          useControllerProps={{ control, name: 'dockerfile' }}
                          formFieldProps={{
                            slotLabel: 'Dockerfile',
                            slotHelp: asks('dockerfile')
                              ? 'Several were found — name the one that builds the agent.'
                              : 'Detected. Edit to override.',
                          }}
                        />
                      )}
                      {derived && (
                        <ControlledTextInput
                          useControllerProps={{ control, name: 'startCommand' }}
                          formFieldProps={{
                            slotLabel: 'Start Command',
                            slotHelp: asks('start_command')
                              ? 'The ENTRYPOINT is a shell form, so it cannot be read. Must be absolute — ' +
                                'the sandbox replaces PATH.'
                              : 'Derived from the Dockerfile. Edit to override.',
                          }}
                        />
                      )}
                      {derived && (
                        <ControlledTextInput
                          useControllerProps={{ control, name: 'binaries' }}
                          formFieldProps={{
                            slotLabel: 'Interpreter Globs',
                            slotHelp:
                              'Comma-separated. Scopes which processes may egress; a glob matching no ' +
                              'process grants nothing while looking like it grants something.',
                          }}
                        />
                      )}
                      {derived && (
                        <ControlledSelect
                          useControllerProps={{ control, name: 'harness' }}
                          items={HARNESS_ITEMS}
                          formFieldProps={{
                            slotLabel: 'Harness',
                            slotHelp:
                              'Not readable from the project. Decides whether a guardrail can refuse a ' +
                              'tool call at all.',
                          }}
                        />
                      )}
                      {derived && (
                        <label className="flex items-center gap-2" data-testid="relay-confirmed">
                          <input
                            type="checkbox"
                            checked={relayConfirmed}
                            onChange={(event) => setRelayConfirmed(event.target.checked)}
                          />
                          <Text kind="body/regular/sm">
                            NeMo Relay is attached to this agent (middleware + plugin.initialize()).
                            Without it the victim emits no telemetry and the run cannot be scored.
                          </Text>
                        </label>
                      )}
                    </Stack>
                  )}
                  <ControlledTextInput
                    useControllerProps={{ control, name: 'egress' }}
                    formFieldProps={{
                      slotLabel: 'Egress Allow-list',
                      slotHelp: inspectAgent.isPending
                        ? 'Detecting from the agent…'
                        : 'Comma-separated host[:port]. Auto-detected from the agent config (model ' +
                          'endpoints, network MCP servers). Add hosts only the packaged code knows about.',
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
          </Stack>
        </Panel>
      </Stack>
    </AccessibleTitle>
  );
};
