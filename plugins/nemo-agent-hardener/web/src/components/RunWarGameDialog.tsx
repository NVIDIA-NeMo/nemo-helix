// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The "Start war-game" dialog: per-run config plus the attack/benign sources for a single launch.
 *
 * Owns its own form state rather than taking ~20 props. That also hardens the seeding rule this
 * dialog needs: values are copied from the manifest exactly once per opening, so a refetch while it
 * is open — "Save as default" invalidates the query — cannot wipe what the user typed.
 *
 * `Start` never writes the manifest and `Save as default` never launches; the parent owns both
 * mutations and this component only reports a validated config.
 */

import { useUploadBenignSuiteFileset, useUploadHitlogFileset } from '@agent-hardener/api/filesets';
import { ModelGroupFields } from '@agent-hardener/components/ModelGroupFields';
import {
  useAgentHardenerGetModelConfigDefaults,
  useAgentHardenerListRuns,
} from '@agent-hardener/generated/api';
import type { AgentHardenerManifest, AgentHardenerRun, WarGameModels } from '@agent-hardener/generated/schema';
import { useToast } from '@agent-hardener/host';
import {
  BENIGN_SOURCE_LABEL,
  INTENSITY_LABEL,
  REPLAY_SOURCE_LABEL,
} from '@agent-hardener/routes/AgentHardenerManifestDetailRoute/constants';
import type {
  AttackIntensity,
  BenignSource,
  DefenderSelection,
  ReplaySource,
} from '@agent-hardener/routes/AgentHardenerManifestDetailRoute/types';
import { FEEDBACK } from '@agent-hardener/theme';
import { AccordionSection, FileUpload, FormModal } from '@nemo/common';
import {
  AccordionRoot,
  Button,
  Checkbox,
  Flex,
  FormField,
  SegmentedControl,
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectRoot,
  SelectTrigger,
  Stack,
  Text,
  TextInput,
} from '@nvidia/foundations-react-core';
import { FC, useEffect, useRef, useState } from 'react';

/** The config half — what a run overrides, and what "save as default" persists. */
export interface RunConfig {
  defenders: ('guardrails' | 'openshell')[];
  attack_intensity: AttackIntensity;
  rounds: number;
  models: WarGameModels;
  port?: number;
}

/** A launch: the config plus the per-run-only attack and benign sources. */
export interface RunLaunch extends RunConfig {
  replay_hitlog_fileset?: string;
  benign_suite_fileset?: string;
}

export interface RunWarGameDialogProps {
  open: boolean;
  onClose: () => void;
  workspace: string;
  manifestName: string;
  /** Seeds the form on each opening; per-run tweaks stay ephemeral. */
  manifest?: AgentHardenerManifest;
  starting: boolean;
  savingDefault: boolean;
  onStart: (launch: RunLaunch) => void;
  onSaveDefault: (config: RunConfig) => void;
}

export const RunWarGameDialog: FC<RunWarGameDialogProps> = ({
  open,
  onClose,
  workspace,
  manifestName,
  manifest,
  starting,
  savingDefault,
  onStart,
  onSaveDefault,
}) => {
  const toast = useToast();
  const { data: modelDefaults } = useAgentHardenerGetModelConfigDefaults(workspace, { query: {} });

  const [port, setPort] = useState('');
  const [defenders, setDefenders] = useState<DefenderSelection>({
    guardrails: true,
    openshell: true,
  });
  const [intensity, setIntensity] = useState<AttackIntensity>('standard');
  const [rounds, setRounds] = useState('1');
  const [models, setModels] = useState<WarGameModels>({});
  const [mode, setMode] = useState<'live' | 'replay'>('live');
  const [replaySource, setReplaySource] = useState<ReplaySource>('last');
  const [hitlogFile, setHitlogFile] = useState<File | undefined>();
  const [hitlogFileset, setHitlogFileset] = useState<string | undefined>();
  const [benignSource, setBenignSource] = useState<BenignSource>('manifest');
  const [benignFile, setBenignFile] = useState<File | undefined>();
  const [benignFileset, setBenignFileset] = useState<string | undefined>();

  // Read through a ref so the seed effect depends on `open` alone: `manifest` gets a new identity on
  // every refetch, and re-seeding mid-edit is the exact bug this guards against.
  const manifestRef = useRef(manifest);
  manifestRef.current = manifest;
  const seeded = useRef(false);
  useEffect(() => {
    if (!open) {
      seeded.current = false;
      return;
    }
    if (seeded.current) return;
    seeded.current = true;
    const m = manifestRef.current;
    setPort(m?.port ? String(m.port) : '');
    // Empty stored list means agent-hardener defaults (all defenders) — reflect that as both checked.
    const saved = m?.defenders ?? [];
    setDefenders(
      saved.length
        ? { guardrails: saved.includes('guardrails'), openshell: saved.includes('openshell') }
        : { guardrails: true, openshell: true }
    );
    setIntensity((m?.attack_intensity as AttackIntensity | undefined) ?? 'standard');
    setRounds(m?.rounds ? String(m.rounds) : '1');
    setModels(m?.models ?? {});
    setMode('live');
    setReplaySource('last');
    setHitlogFile(undefined);
    setHitlogFileset(undefined);
    setBenignSource('manifest');
    setBenignFile(undefined);
    setBenignFileset(undefined);
  }, [open]);

  // "Last run" replay source: the newest prior run of THIS manifest that saved a hitlog fileset.
  // Fetched only when that source is actually selected.
  const { data: runsData } = useAgentHardenerListRuns(
    workspace,
    { sort: '-created_at', page_size: 20, filter: { manifest_id: manifestName } },
    { query: { enabled: open && mode === 'replay' && replaySource === 'last' } }
  );
  const lastHitlogFileset = ((runsData?.data ?? []) as AgentHardenerRun[]).find(
    (r) => r.hitlog_fileset
  )?.hitlog_fileset;

  const uploadHitlog = useUploadHitlogFileset();
  const onHitlogDrop = async (accepted: File[]): Promise<void> => {
    const file = accepted[0];
    if (!file) return;
    setHitlogFile(file);
    setHitlogFileset(undefined);
    try {
      setHitlogFileset(await uploadHitlog.mutateAsync({ workspace, manifestName, file }));
    } catch {
      toast.error('Failed to upload the hitlog file.');
    }
  };

  const uploadBenign = useUploadBenignSuiteFileset();
  const onBenignDrop = async (accepted: File[]): Promise<void> => {
    const file = accepted[0];
    if (!file) return;
    setBenignFile(file);
    setBenignFileset(undefined);
    try {
      setBenignFileset(await uploadBenign.mutateAsync({ workspace, manifestName, file }));
    } catch {
      toast.error('Failed to upload the benign suite file.');
    }
  };

  /** The config both actions share; `null` when the selection is not launchable. */
  const buildRunConfig = (): RunConfig | null => {
    const keys = (['guardrails', 'openshell'] as const).filter((k) => defenders[k]);
    if (keys.length === 0) {
      toast.error('Select at least one defender.');
      return null;
    }
    return {
      defenders: keys,
      attack_intensity: intensity,
      rounds: Number(rounds) || 1,
      models,
      ...(port ? { port: Number(port) } : {}),
    };
  };

  const start = () => {
    const cfg = buildRunConfig();
    if (!cfg) return;
    let replay_hitlog_fileset: string | undefined;
    if (mode === 'replay') {
      replay_hitlog_fileset = replaySource === 'upload' ? hitlogFileset : lastHitlogFileset;
      if (!replay_hitlog_fileset) {
        toast.error(
          replaySource === 'upload'
            ? 'Upload a hitlog file to replay first.'
            : 'No previous run of this manifest has recorded hits — run a live attack once, or upload a hitlog.'
        );
        return;
      }
    }
    // Uploaded benign suite (if chosen) overrides the manifest's suite for this run.
    if (benignSource === 'upload' && !benignFileset) {
      toast.error('Upload a benign suite CSV first.');
      return;
    }
    onStart({
      ...cfg,
      ...(replay_hitlog_fileset ? { replay_hitlog_fileset } : {}),
      ...(benignSource === 'upload' && benignFileset
        ? { benign_suite_fileset: benignFileset }
        : {}),
    });
  };

  const saveAsDefault = () => {
    const cfg = buildRunConfig();
    if (cfg) onSaveDefault(cfg);
  };

  return (
    <FormModal
      open={open}
      title="Start war-game"
      submitButtonText="Start"
      loading={starting}
      submitDisabled={
        (mode === 'replay' &&
          (replaySource === 'upload'
            ? !hitlogFileset || uploadHitlog.isPending
            : !lastHitlogFileset)) ||
        (benignSource === 'upload' && (!benignFileset || uploadBenign.isPending))
      }
      onSubmit={(e) => {
        e.preventDefault();
        start();
      }}
      onClose={onClose}
    >
      <Stack gap="density-md">
        <Text kind="body/regular/sm" className="text-subtle">
          Config applies to this run only. Use “Save as default” to make it the manifest baseline.
        </Text>

        <FormField
          name="port"
          slotLabel="Victim Port"
          slotHelp="Port the war-game targets on the victim agent."
        >
          <TextInput
            value={port}
            onChange={(e) => setPort(e.target.value.replace(/[^0-9]/g, ''))}
          />
        </FormField>

        <Stack gap="density-sm">
          <Text kind="body/semibold/sm" className="uppercase tracking-wide text-subtle">
            Defenders
          </Text>
          <Flex align="center" gap="density-sm">
            <Checkbox
              checked={defenders.guardrails}
              onCheckedChange={(c) => setDefenders((d) => ({ ...d, guardrails: c === true }))}
              attributes={{ CheckboxInput: { 'aria-label': 'Guardrails defender' } }}
            />
            <Text kind="body/regular/sm">Guardrails defender</Text>
          </Flex>
          <Flex align="center" gap="density-sm">
            <Checkbox
              checked={defenders.openshell}
              onCheckedChange={(c) => setDefenders((d) => ({ ...d, openshell: c === true }))}
              attributes={{ CheckboxInput: { 'aria-label': 'OpenShell policy defender' } }}
            />
            <Text kind="body/regular/sm">OpenShell policy defender</Text>
          </Flex>
        </Stack>

        <FormField
          name="intensity"
          slotLabel="Attack Intensity"
          slotHelp="How hard the garak attacker probes the agent — more probes and generations at higher levels."
        >
          {/* The trigger echoes the raw value, so use capitalized values (== display) and lowercase for the API. */}
          <SelectRoot
            value={INTENSITY_LABEL[intensity]}
            onValueChange={(v: string) => setIntensity(v.toLowerCase() as AttackIntensity)}
          >
            <SelectTrigger className="w-full" placeholder="Select intensity" />
            <SelectContent className="w-(--radix-popper-anchor-width)">
              <SelectListbox>
                <SelectItem value="Light">Light</SelectItem>
                <SelectItem value="Standard">Standard</SelectItem>
                <SelectItem value="Thorough">Thorough</SelectItem>
              </SelectListbox>
            </SelectContent>
          </SelectRoot>
        </FormField>

        <FormField
          name="rounds"
          slotLabel="Rounds"
          slotHelp="Iterative attack → defend → validate → redeploy cycles. More rounds go deeper but take longer."
        >
          <TextInput
            value={rounds}
            onChange={(e) => setRounds(e.target.value.replace(/[^0-9]/g, ''))}
          />
        </FormField>

        <AccordionRoot>
          <AccordionSection value="models" title="Models (optional)">
            <ModelGroupFields
              value={models}
              onChange={setModels}
              workspace={workspace}
              defaults={modelDefaults}
            />
          </AccordionSection>
        </AccordionRoot>

        <FormField
          name="benignSource"
          slotLabel="Benign suite"
          slotHelp="The benign requests replayed after hardening to confirm the agent still works. Defaults to the manifest's suite; upload a requests.csv to override it for this run."
        >
          <SelectRoot
            value={BENIGN_SOURCE_LABEL[benignSource]}
            onValueChange={(v: string) =>
              setBenignSource(v === BENIGN_SOURCE_LABEL.upload ? 'upload' : 'manifest')
            }
          >
            <SelectTrigger className="w-full" placeholder="Select benign suite" />
            <SelectContent className="w-(--radix-popper-anchor-width)">
              <SelectListbox>
                <SelectItem value={BENIGN_SOURCE_LABEL.manifest}>
                  {BENIGN_SOURCE_LABEL.manifest}
                </SelectItem>
                <SelectItem value={BENIGN_SOURCE_LABEL.upload}>
                  {BENIGN_SOURCE_LABEL.upload}
                </SelectItem>
              </SelectListbox>
            </SelectContent>
          </SelectRoot>
        </FormField>

        {benignSource === 'upload' ? (
          <FileUpload
            label="Benign suite"
            accept={{ 'text/csv': ['.csv'] }}
            multiple={false}
            files={benignFile ? [benignFile] : []}
            onDropAccepted={(accepted) => void onBenignDrop(accepted)}
            onRemoveFile={() => {
              setBenignFile(undefined);
              setBenignFileset(undefined);
            }}
            helperText={
              uploadBenign.isPending
                ? 'Uploading…'
                : benignFileset
                  ? 'Uploaded — will override the manifest suite for this run.'
                  : 'A benign requests.csv (tool,payload,label,rationale,persona).'
            }
          />
        ) : null}

        <FormField
          name="mode"
          slotLabel="Attack mode"
          slotHelp="Live runs a fresh garak attack; Replay skips it and replays recorded hits against the defended agent."
        >
          <SegmentedControl
            className="w-full"
            value={mode}
            onValueChange={(v) => setMode(v as 'live' | 'replay')}
            items={[
              { value: 'live', children: 'Live attack' },
              { value: 'replay', children: 'Replay recorded hits' },
            ]}
          />
        </FormField>

        {mode === 'replay' ? (
          <>
            <FormField name="replaySource" slotLabel="Hits to replay">
              <SelectRoot
                value={REPLAY_SOURCE_LABEL[replaySource]}
                onValueChange={(v: string) =>
                  setReplaySource(v === REPLAY_SOURCE_LABEL.upload ? 'upload' : 'last')
                }
              >
                <SelectTrigger className="w-full" placeholder="Select hits to replay" />
                <SelectContent className="w-(--radix-popper-anchor-width)">
                  <SelectListbox>
                    <SelectItem value={REPLAY_SOURCE_LABEL.last}>
                      {REPLAY_SOURCE_LABEL.last}
                    </SelectItem>
                    <SelectItem value={REPLAY_SOURCE_LABEL.upload}>
                      {REPLAY_SOURCE_LABEL.upload}
                    </SelectItem>
                  </SelectListbox>
                </SelectContent>
              </SelectRoot>
            </FormField>

            {replaySource === 'last' ? (
              <Text
                kind="body/regular/sm"
                className={lastHitlogFileset ? 'text-subtle' : undefined}
                style={lastHitlogFileset ? undefined : { color: FEEDBACK.warning }}
              >
                {lastHitlogFileset
                  ? "Replays this manifest's most recent recorded hits."
                  : 'No previous run of this manifest has recorded hits — run a live attack once, or upload a hitlog.'}
              </Text>
            ) : (
              <FileUpload
                label="Hitlog"
                accept={{ 'application/jsonl': ['.jsonl', '.json'] }}
                multiple={false}
                files={hitlogFile ? [hitlogFile] : []}
                onDropAccepted={(accepted) => void onHitlogDrop(accepted)}
                onRemoveFile={() => {
                  setHitlogFile(undefined);
                  setHitlogFileset(undefined);
                }}
                helperText={
                  uploadHitlog.isPending
                    ? 'Uploading…'
                    : hitlogFileset
                      ? 'Uploaded — ready to replay.'
                      : 'A garak hitlog (.jsonl) recording the attack hits to replay.'
                }
              />
            )}
          </>
        ) : null}

        <Flex>
          <Button kind="tertiary" type="button" disabled={savingDefault} onClick={saveAsDefault}>
            Save as default
          </Button>
        </Flex>
      </Stack>
    </FormModal>
  );
};
