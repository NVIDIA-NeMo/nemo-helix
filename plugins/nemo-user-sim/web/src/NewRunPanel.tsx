// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Button,
  Card,
  Checkbox,
  ComboboxContent,
  ComboboxItem,
  ComboboxListbox,
  ComboboxRoot,
  ComboboxTrigger,
  Flex,
  FormField,
  SegmentedControl,
  SidePanel,
  Skeleton,
  Slider,
  Spinner,
  Stack,
  Tag,
  Text,
  TextInput,
} from '@nvidia/foundations-react-core';
import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router';
import {
  useAgents,
  useCreateJob,
  useModels,
  usePersonaPreview,
  useProbes,
  useProbeSuggestions,
  useScenarioSets,
} from './api';
import { useHost } from './host';
import { getRunRoute } from './paths';
import { styleColor, words } from './report';
import type { PersonaCard, PersonaPreview, ScenarioSet, SimulateRequest } from './types';

const TARGET_KINDS = [
  { value: 'agent', children: 'Agent' },
  { value: 'model', children: 'Model' },
];

const BUILT_IN = 'built-in';
const setLabel = (set: ScenarioSet) =>
  `${set.fileset.replace(/^user-sim-scenarios-/, '')} · ${set.size ? `${set.size} topics` : set.kind}`;

const SIZES = [
  { value: '10', children: 'Small · 10' },
  { value: '50', children: 'Medium · 50' },
  { value: '200', children: 'Large · 200' },
];

const PersonaRow = ({
  persona,
  removed,
  disabled,
  onToggle,
}: {
  persona: PersonaCard;
  removed: boolean;
  disabled: boolean;
  onToggle: () => void;
}) => (
  <Card density="compact" className={removed ? 'opacity-50' : undefined}>
    <Flex justify="between" align="center" gap="density-md">
      <Stack gap="0">
        <Flex align="center" gap="density-sm">
          <Text kind="body/semibold/sm">{persona.name}</Text>
          <Tag kind="solid" color={styleColor(persona.style)} density="compact" readOnly>
            {persona.style}
          </Tag>
        </Flex>
        <Text kind="body/regular/xs" className="text-subtle">
          {persona.age} · {words(persona.education)} · {words(persona.occupation)}
          {persona.location ? ` · ${persona.location}` : ''}
        </Text>
        <Text kind="body/regular/xs" className="text-subtle">
          {words(persona.probe)} · patience {persona.traits.patience ?? '–'} · tech literacy{' '}
          {persona.traits.tech_literacy ?? '–'}
        </Text>
      </Stack>
      <Button kind="tertiary" size="small" disabled={disabled} onClick={onToggle}>
        {removed ? 'Restore' : 'Remove'}
      </Button>
    </Flex>
  </Card>
);

/**
 * Start a run: pick what to test and, optionally, an uploaded topic set, take or adjust the probes, then
 * preview who it will meet. Given an earlier run, it starts from that run's settings and keeps its custom
 * scenarios or topics. Only the CLI uploads a set; Studio picks one that's already there.
 */
export const NewRunPanel = ({
  open,
  onClose,
  from,
}: {
  open: boolean;
  onClose: () => void;
  from?: { job: string; spec: SimulateRequest };
}) => {
  const { workspaceId, notifications } = useHost();
  const navigate = useNavigate();
  const [kind, setKind] = useState<'agent' | 'model'>(from?.spec.target.kind ?? 'agent');
  const [target, setTarget] = useState<string | null>(
    from ? (from.spec.target.agent ?? from.spec.target.model ?? null) : null
  );
  const [chosen, setChosen] = useState<string[] | null>(from?.spec.probes ?? null);
  const [users, setUsers] = useState(String(from?.spec.users ?? 10));
  const [turns, setTurns] = useState(from?.spec.max_turns ?? 5);
  const [locale, setLocale] = useState(from?.spec.locale ?? 'en_US');
  const [scenarios, setScenarios] = useState<string | null>(from?.spec.scenarios ?? null);
  // A run made before this setting existed has no value: start off.
  const [ownTools, setOwnTools] = useState<boolean>(from?.spec.target.use_agent_tools ?? false);
  const [removed, setRemoved] = useState<Set<string>>(new Set());
  const [allProbes, setAllProbes] = useState(false);
  // Kept apart from the mutation, which clears its data while a refill is picking.
  const [picked, setPicked] = useState<PersonaPreview | null>(null);
  const agents = useAgents();
  const models = useModels();
  const { data: catalog = [] } = useProbes();
  const suggestions = useProbeSuggestions(kind === 'agent' ? target : null);
  const sets = useScenarioSets();
  const preview = usePersonaPreview();
  const create = useCreateJob<SimulateRequest>('simulate');

  const reasons = useMemo(
    () => new Map((suggestions.data?.probes ?? []).map((probe) => [probe.name, probe.reason])),
    [suggestions.data]
  );
  const set = (sets.data ?? []).find((each) => each.fileset === scenarios);
  // Until someone changes the selection, the probes are the ones the chosen set runs in, or else the
  // ones inferred from the agent.
  const probes = chosen ?? set?.probes ?? [...reasons.keys()];
  const ordered = [...catalog].sort(
    (a, b) => Number(reasons.has(b.name)) - Number(reasons.has(a.name))
  );
  // With an agent picked, show what's suggested or chosen and fold the rest away.
  const focused = kind === 'agent' && !!target && !allProbes;
  const shown = focused
    ? ordered.filter((probe) => reasons.has(probe.name) || probes.includes(probe.name))
    : ordered;
  // The combobox filters these as the user types and shows them all again on reopen.
  const options =
    kind === 'agent' ? (agents.data ?? []).map((agent) => agent.name) : (models.data ?? []);
  const listing = kind === 'agent' ? agents : models;
  const spec: SimulateRequest = {
    target: {
      kind,
      [kind]: target ?? '',
      ...(kind === 'agent' && ownTools ? { use_agent_tools: true } : {}),
    },
    probes,
    users: Number(users),
    locale: locale.trim(),
    max_turns: turns,
    ...(scenarios ? { scenarios } : {}),
  };
  const ready = !!target && probes.length > 0 && Number(users) >= 1 && Number(users) <= 2000;

  const choose = (next: string | null) => {
    setTarget(next);
    setChosen(null);
    setAllProbes(false);
    setPicked(null);
  };
  const chooseSet = (next: string) => {
    setScenarios(next === BUILT_IN ? null : next);
    setChosen(null);
    setPicked(null);
  };
  const toggleProbe = (name: string, on: boolean) =>
    setChosen(on ? [...probes, name] : probes.filter((probe) => probe !== name));
  const toggleRemoved = (uuid: string) =>
    setRemoved((current) => {
      const next = new Set(current);
      if (next.has(uuid)) next.delete(uuid);
      else next.add(uuid);
      return next;
    });
  const start = (body: SimulateRequest) =>
    create.mutate(body, {
      onSuccess: (job) => {
        onClose();
        navigate(getRunRoute(workspaceId, job.name));
      },
      onError: (error) => notifications.notify(`Couldn't start the run. ${error.message}`, 'error'),
    });
  // With a preview showing, this keeps its personas minus the removed ones and picks the rest anew.
  const pick = (body: SimulateRequest) =>
    preview.mutate(body, {
      onSuccess: (next) => {
        setPicked(next);
        setRemoved(new Set());
      },
      onError: (error) => notifications.notify(`Couldn't pick personas. ${error.message}`, 'error'),
    });

  const personas = picked?.personas ?? [];
  const kept = personas.length - removed.size;
  const missing = Number(users) - kept;
  const fromPreview = picked ? { preview: picked.preview, exclude_personas: [...removed] } : {};

  const footer = picked ? (
    <Flex justify="between" gap="density-sm" className="w-full">
      <Button kind="tertiary" disabled={preview.isPending} onClick={() => setPicked(null)}>
        Back
      </Button>
      <Flex gap="density-sm">
        {missing > 0 ? (
          <Button
            kind="secondary"
            disabled={preview.isPending}
            onClick={() => pick({ ...spec, ...fromPreview })}
          >
            {preview.isPending ? <Spinner kind="ring" aria-label="Picking personas" /> : null}
            {preview.isPending ? `Picking ${missing} more…` : `Pick ${missing} more`}
          </Button>
        ) : null}
        <Button
          disabled={kept === 0 || preview.isPending || create.isPending}
          onClick={() => start({ ...spec, ...fromPreview })}
        >
          Run with {kept} {kept === 1 ? 'persona' : 'personas'}
        </Button>
      </Flex>
    </Flex>
  ) : (
    <Flex justify="end" gap="density-sm" className="w-full">
      <Button kind="tertiary" disabled={!ready || create.isPending} onClick={() => start(spec)}>
        Start without preview
      </Button>
      <Button disabled={!ready || preview.isPending} onClick={() => pick(spec)}>
        {preview.isPending ? <Spinner kind="ring" aria-label="Picking personas" /> : null}
        {preview.isPending ? `Picking ${users} personas…` : 'Preview personas'}
      </Button>
    </Flex>
  );

  return (
    <SidePanel
      open={open}
      onOpenChange={(next) => !next && onClose()}
      side="right"
      bordered
      modal
      className="w-[560px]"
      slotHeading={picked ? 'Who this run will meet' : from ? 'Run again' : 'New simulation'}
      slotFooter={footer}
    >
      {picked ? (
        <Stack gap="density-sm">
          <Text kind="body/regular/sm" className="text-subtle">
            Picked from census-grounded personas. Remove any that don&apos;t fit, then run with the
            personas left or pick more to replace them.
          </Text>
          {personas.map((persona) =>
            preview.isPending && removed.has(persona.persona_uuid) ? null : (
              <PersonaRow
                key={persona.persona_uuid}
                persona={persona}
                removed={removed.has(persona.persona_uuid)}
                disabled={preview.isPending}
                onToggle={() => toggleRemoved(persona.persona_uuid)}
              />
            )
          )}
          {preview.isPending
            ? Array.from({ length: missing }, (_, index) => (
                <Card key={index} density="compact">
                  <Stack gap="density-xs">
                    <Skeleton className="w-1/3" />
                    <Skeleton className="w-2/3" />
                  </Stack>
                </Card>
              ))
            : null}
        </Stack>
      ) : preview.isPending ? (
        <Flex justify="center" className="py-16">
          <Spinner slotDescription={`Picking ${users} personas. This takes about 20 seconds.`} />
        </Flex>
      ) : (
        <Stack gap="density-lg">
          {from ? (
            <Text kind="body/regular/sm" className="text-subtle">
              Same settings as {from.job}
              {scenarios ? `, with its topics from ${scenarios}` : ''}. New simulated users each time.
            </Text>
          ) : null}
          <FormField name="target" slotLabel="What to test">
            <Stack gap="density-sm">
              <SegmentedControl
                className="w-full"
                items={TARGET_KINDS}
                value={kind}
                onValueChange={(value) => {
                  setKind(value as 'agent' | 'model');
                  choose(null);
                }}
              />
              {/* Keyed by kind so switching Agent and Model starts a fresh field. */}
              <ComboboxRoot
                key={kind}
                value={target ?? ''}
                onValueChange={(value) => {
                  if (typeof value === 'string') choose(value || null);
                }}
              >
                <ComboboxTrigger
                  placeholder={kind === 'agent' ? 'Search agents…' : 'Search models…'}
                />
                <ComboboxContent>
                  <ComboboxListbox
                    emptyStateMessage={
                      listing.isPending
                        ? `Loading ${kind}s…`
                        : listing.isError
                          ? `Couldn't load ${kind}s. ${listing.error.message}`
                          : `No ${kind}s match`
                    }
                  >
                    {options.map((name) => (
                      <ComboboxItem key={name} value={name}>
                        {name}
                      </ComboboxItem>
                    ))}
                  </ComboboxListbox>
                </ComboboxContent>
              </ComboboxRoot>
            </Stack>
          </FormField>

          {kind === 'agent' ? (
            <FormField name="agent-tools" slotLabel="Agent's tools">
              <Flex align="start" gap="density-sm">
                <Checkbox
                  checked={ownTools}
                  onCheckedChange={(checked) => setOwnTools(checked === true)}
                  attributes={{ CheckboxInput: { 'aria-label': "Use the agent's own tools" } }}
                />
                <Stack gap="0">
                  <Text kind="body/regular/sm">Use the agent&apos;s own tools</Text>
                  <Text kind="body/regular/xs" className="text-subtle">
                    Only turn this on if they&apos;re safe to call for real, like search.
                  </Text>
                  <Text kind="body/regular/xs" className="text-subtle">
                    Off: the agent only gets UserSim&apos;s simulated tools.
                  </Text>
                </Stack>
              </Flex>
            </FormField>
          ) : null}

          {from ? null : (
            <FormField
              name="topics"
              slotLabel="Topics"
              slotHelp="What the simulated users ask about. Upload your own with nemo user-sim run --scenarios."
            >
              <ComboboxRoot
                value={scenarios ?? BUILT_IN}
                onValueChange={(value) => {
                  if (typeof value === 'string') chooseSet(value || BUILT_IN);
                }}
              >
                <ComboboxTrigger placeholder="UserSim's built-in topics" />
                <ComboboxContent>
                  <ComboboxListbox
                    emptyStateMessage={
                      sets.isError ? `Couldn't load topic sets. ${sets.error.message}` : 'No topic sets match'
                    }
                  >
                    <ComboboxItem value={BUILT_IN}>UserSim&apos;s built-in topics</ComboboxItem>
                    {(sets.data ?? []).map((each) => (
                      <ComboboxItem key={each.fileset} value={each.fileset}>
                        {setLabel(each)}
                      </ComboboxItem>
                    ))}
                  </ComboboxListbox>
                </ComboboxContent>
              </ComboboxRoot>
            </FormField>
          )}

          <FormField
            name="probes"
            slotLabel="Probes"
            slotHelp={
              kind === 'agent'
                ? 'Chosen for you from the agent. Change them if you like.'
                : 'What the simulated users try to do.'
            }
          >
            <Stack gap="density-sm">
              {suggestions.isFetching ? (
                <Text kind="body/regular/sm" className="text-subtle">
                  Reading {target}&apos;s config and ETHOS.md…
                </Text>
              ) : suggestions.data ? (
                <Text kind="body/regular/xs" className="text-subtle">
                  Inferred from {target}&apos;s {suggestions.data.read.join(' and ')}.
                </Text>
              ) : null}
              {shown.map((probe) => (
                <Flex key={probe.name} align="start" gap="density-sm">
                  <Checkbox
                    checked={probes.includes(probe.name)}
                    onCheckedChange={(checked) => toggleProbe(probe.name, checked === true)}
                    attributes={{ CheckboxInput: { 'aria-label': probe.label } }}
                  />
                  <Stack gap="0">
                    <Flex align="center" gap="density-sm">
                      <Text kind="body/regular/sm">{probe.label}</Text>
                      {reasons.has(probe.name) ? (
                        <Tag kind="solid" color="green" readOnly>
                          Suggested
                        </Tag>
                      ) : null}
                    </Flex>
                    <Text kind="body/regular/xs" className="text-subtle">
                      {reasons.get(probe.name) ?? probe.tests}
                    </Text>
                  </Stack>
                </Flex>
              ))}
              {focused && shown.length < ordered.length ? (
                <Button kind="tertiary" size="small" onClick={() => setAllProbes(true)}>
                  More probes ({ordered.length - shown.length})
                </Button>
              ) : null}
            </Stack>
          </FormField>

          <FormField name="users" slotLabel="Size" slotHelp="How many simulated users, up to 2000.">
            <Stack gap="density-sm">
              <SegmentedControl
                className="w-full"
                items={SIZES}
                value={SIZES.some((size) => size.value === users) ? users : ''}
                onValueChange={setUsers}
              />
              <TextInput
                value={users}
                onChange={(event) => setUsers(event.target.value.replace(/[^0-9]/g, ''))}
              />
            </Stack>
          </FormField>

          <FormField
            name="turns"
            slotLabel={`Conversation length: ${turns} user turns`}
            slotHelp="How many times each simulated user speaks before the conversation ends."
          >
            <Slider
              value={turns}
              min={2}
              max={10}
              step={1}
              onValueChange={(value) => setTurns(Array.isArray(value) ? value[0] : value)}
              aria-label="Conversation length"
            />
          </FormField>

          <FormField
            name="locale"
            slotLabel="Persona locale"
            slotHelp="Personas for this locale must be installed with nemo user-sim setup."
          >
            <TextInput value={locale} onChange={(event) => setLocale(event.target.value)} />
          </FormField>
        </Stack>
      )}
    </SidePanel>
  );
};
