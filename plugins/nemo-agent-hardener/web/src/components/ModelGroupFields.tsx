// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Model selection for a war-game — the three groups (attack / analysis / agent), each pre-filled with
 * its built-in default. Controlled via `value`/`onChange` so it drops into both the react-hook-form
 * create wizards and the useState run-launch dialog. For a custom endpoint a user can set a base URL
 * and pick (or create) a Secret for the key, then "Test connection" to confirm reachability — which
 * lists the models those credentials can actually reach.
 */

import { usePlatformSdk } from '@agent-hardener/api/platform';
import { useAgentHardenerValidateModelConfig } from '@agent-hardener/generated/api';
import type {
  ModelChoice,
  ModelConfigDefaults,
  WarGameModels,
} from '@agent-hardener/generated/schema';
import { useToast } from '@agent-hardener/host';
import { CreateSecretModal, getErrorMessage } from '@nemo/common';
import {
  Button,
  Flex,
  FormField,
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectRoot,
  SelectTrigger,
  Stack,
  Text,
  TextInput,
} from '@nvidia/foundations-react-core';
import { useQuery } from '@tanstack/react-query';
import { FC, PropsWithChildren, useState } from 'react';
import { createPortal } from 'react-dom';

/** The two credentialed groups the user can point at a custom endpoint (the agent model routes via the IGW). */
type CredentialedGroup = 'attack' | 'analysis';

const GROUP_LABEL: Record<keyof WarGameModels, string> = {
  attack: 'Attack model',
  analysis: 'Analysis model',
  safety: 'Safety model',
};
const GROUP_HELP: Record<keyof WarGameModels, string> = {
  attack: 'The garak red-team + detector models that probe the agent.',
  analysis:
    'The defenders and the benign validator — both its suite generation (synth) and judging — one shared model.',
  safety:
    "The LLM the generated guardrail uses to screen the agent's traffic. Leave blank to reuse the agent's " +
    'own model, which makes guardrail quality depend on the model being attacked.',
};

const NO_SECRET = '';

export interface ModelGroupFieldsProps {
  value: WarGameModels;
  onChange: (next: WarGameModels) => void;
  workspace: string;
  /** Built-in defaults (from `useAgentHardenerGetModelConfigDefaults`) shown as placeholders. */
  defaults?: ModelConfigDefaults;
}

/** Merge a partial change into one group, dropping the group to `null` when it becomes empty. */
function withGroup(
  models: WarGameModels,
  group: keyof WarGameModels,
  patch: Partial<ModelChoice>
): WarGameModels {
  const merged: ModelChoice = { ...(models[group] ?? {}), ...patch };
  const isEmpty = !merged.model && !merged.base_url && !merged.api_key_secret;
  return { ...models, [group]: isEmpty ? undefined : merged };
}

/** One model group as a visually distinct block: a heading + help, a top divider on all but the first. */
const GroupSection: FC<PropsWithChildren<{ label: string; help: string; divider?: boolean }>> = ({
  label,
  help,
  divider,
  children,
}) => (
  <Stack gap="density-sm" className={divider ? 'border-t border-base pt-4' : undefined}>
    <div>
      <Text kind="body/semibold/sm">{label}</Text>
      <Text kind="body/regular/sm" className="text-subtle">
        {help}
      </Text>
    </div>
    {children}
  </Stack>
);

export const ModelGroupFields: FC<ModelGroupFieldsProps> = ({
  value,
  onChange,
  workspace,
  defaults,
}) => (
  <Stack gap="density-md">
    <CredentialedGroupFields
      group="attack"
      value={value}
      onChange={onChange}
      workspace={workspace}
      defaultModel={defaults?.attack.model}
      defaultBaseUrl={defaults?.attack.base_url}
    />
    <CredentialedGroupFields
      group="analysis"
      value={value}
      onChange={onChange}
      workspace={workspace}
      defaultModel={defaults?.analysis.model}
      defaultBaseUrl={defaults?.analysis.base_url}
      divider
    />
    {/* Safety (guardrail) model: name only — agent-hardener pins the endpoint + key when it writes the
        guardrail into the victim's plugins.toml, so a base URL or secret here would never be read. */}
    <GroupSection label={GROUP_LABEL.safety} help={GROUP_HELP.safety} divider>
      <FormField name="safety-model" slotLabel="Model">
        <TextInput
          value={value.safety?.model ?? ''}
          placeholder="Reuse the agent's own model"
          onChange={(e) =>
            onChange(withGroup(value, 'safety', { model: e.target.value || undefined }))
          }
        />
      </FormField>
    </GroupSection>
  </Stack>
);

interface CredentialedGroupProps {
  group: CredentialedGroup;
  value: WarGameModels;
  onChange: (next: WarGameModels) => void;
  workspace: string;
  defaultModel?: string;
  defaultBaseUrl?: string;
  divider?: boolean;
}

const CredentialedGroupFields: FC<CredentialedGroupProps> = ({
  group,
  value,
  onChange,
  workspace,
  defaultModel,
  defaultBaseUrl,
  divider,
}) => {
  const choice = value[group] ?? {};
  const [createSecretOpen, setCreateSecretOpen] = useState(false);
  const [testResult, setTestResult] = useState<string | null>(null);
  const validate = useAgentHardenerValidateModelConfig();
  const { secretsListSecrets, useSecretsCreateSecret } = usePlatformSdk();
  const toast = useToast();
  const notify = (message: string, type?: 'success' | 'error' | 'info' | 'warning') =>
    toast[type ?? 'info'](message);
  const createSecret = useSecretsCreateSecret();

  const secrets = useQuery({
    queryKey: ['agent-hardener-model-secrets', workspace],
    queryFn: ({ signal }) => secretsListSecrets(workspace, { page: 1, page_size: 100 }, signal),
    enabled: !!workspace,
  });
  const fetchedSecrets = secrets.data?.data.map((s) => s.name) ?? [];
  // Always include the current selection — a just-created secret isn't in the fetched page yet, so without
  // this the Select would hold a value with no matching option and render blank.
  const secretNames =
    choice.api_key_secret && !fetchedSecrets.includes(choice.api_key_secret)
      ? [choice.api_key_secret, ...fetchedSecrets]
      : fetchedSecrets;

  const onTest = async () => {
    setTestResult('Testing…');
    try {
      const res = await validate.mutateAsync({
        workspace,
        data: {
          model: choice.model ?? undefined,
          base_url: choice.base_url || defaultBaseUrl || '',
          api_key_secret: choice.api_key_secret ?? undefined,
        },
      });
      setTestResult(formatVerdict(res.ok, res.reason, res.available ?? [], res.detail));
    } catch {
      setTestResult('Could not reach the validation service.');
    }
  };

  return (
    <GroupSection label={GROUP_LABEL[group]} help={GROUP_HELP[group]} divider={divider}>
      <FormField name={`${group}-model`} slotLabel="Model">
        <TextInput
          value={choice.model ?? ''}
          placeholder={defaultModel ?? 'Default model'}
          onChange={(e) =>
            onChange(withGroup(value, group, { model: e.target.value || undefined }))
          }
        />
      </FormField>

      <FormField
        name={`${group}-base-url`}
        slotLabel="Custom endpoint (optional)"
        slotHelp="OpenAI-compatible base URL; leave blank to use the default NVIDIA endpoint."
      >
        <TextInput
          value={choice.base_url ?? ''}
          placeholder={defaultBaseUrl ?? 'https://…/v1'}
          onChange={(e) =>
            onChange(withGroup(value, group, { base_url: e.target.value || undefined }))
          }
        />
      </FormField>

      <FormField
        name={`${group}-secret`}
        slotLabel="API key secret (optional)"
        slotHelp="A Secret holding the provider key for a custom endpoint."
      >
        <SelectRoot
          value={choice.api_key_secret ?? NO_SECRET}
          onValueChange={(v: string) =>
            v === '__create__'
              ? setCreateSecretOpen(true)
              : onChange(withGroup(value, group, { api_key_secret: v || undefined }))
          }
        >
          <SelectTrigger className="w-full" placeholder="Select a secret (optional)" />
          <SelectContent className="w-(--radix-popper-anchor-width)">
            <SelectListbox>
              <SelectItem value={NO_SECRET}>None</SelectItem>
              {secretNames.map((name) => (
                <SelectItem key={name} value={name}>
                  {name}
                </SelectItem>
              ))}
              <SelectItem value="__create__">+ Create new secret…</SelectItem>
            </SelectListbox>
          </SelectContent>
        </SelectRoot>
      </FormField>

      <Flex align="center" gap="density-sm">
        <Button
          kind="secondary"
          size="small"
          disabled={
            validate.isPending || (!choice.model && !choice.base_url && !choice.api_key_secret)
          }
          onClick={() => void onTest()}
        >
          Test connection
        </Button>
        {testResult && (
          <Text kind="body/regular/sm" className="text-subtle">
            {testResult}
          </Text>
        )}
      </Flex>

      {/* Portal to <body> so the modal's own <form> is never nested inside the host form (create wizard /
          run dialog), which is invalid HTML and breaks the modal. */}
      {createPortal(
        <CreateSecretModal
          open={createSecretOpen}
          onClose={() => setCreateSecretOpen(false)}
          pending={createSecret.isPending}
          errorText={createSecret.error ? getErrorMessage(createSecret.error) : undefined}
          onNotify={notify}
          onCreate={async (data) => {
            const created = await createSecret.mutateAsync({ workspace, data });
            onChange(withGroup(value, group, { api_key_secret: created.name }));
            void secrets.refetch();
            setCreateSecretOpen(false);
          }}
        />,
        document.body
      )}
    </GroupSection>
  );
};

/** Turn a validate verdict into a short, user-facing line (lists reachable models on a name miss). */
function formatVerdict(
  ok: boolean,
  reason?: string,
  available: string[] = [],
  detail?: string
): string {
  if (ok) return 'Connection OK.';
  if (reason === 'auth') return `Credentials rejected${detail ? ` (${detail})` : ''}.`;
  if (reason === 'unreachable') return `Endpoint unreachable${detail ? ` (${detail})` : ''}.`;
  if (reason === 'unknown_model') {
    const list = available.slice(0, 8).join(', ');
    return `Model not found. Reachable: ${list || 'none'}${available.length > 8 ? ', …' : ''}.`;
  }
  return detail || 'Validation failed.';
}
