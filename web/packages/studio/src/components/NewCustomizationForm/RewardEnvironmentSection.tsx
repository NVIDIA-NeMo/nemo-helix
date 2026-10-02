// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KVPair } from '@nemo/common/src/components/KVPair';
import { getEntityReference, getPartsFromReference } from '@nemo/common/src/namedEntity';
import { useFilesListFilesets } from '@nemo/sdk/generated/platform/files';
import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import {
  Anchor,
  Banner,
  Block,
  FormField,
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectRoot,
  SelectTrigger,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { FormSection } from '@studio/components/NewCustomizationForm/FormSection';
import { LINK_DOCS_GRPO_TRAINING } from '@studio/constants/links';
import {
  type GymEnvironmentManifest,
  useGymEnvironmentManifest,
} from '@studio/hooks/useGymEnvironmentManifest';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getHumanReadableFileSize } from '@studio/util/files';
import type { CustomizationFormFields } from '@studio/util/forms/customization';
import { FC, useMemo } from 'react';
import { useController, useFormContext } from 'react-hook-form';

const GymEnvironmentInfo: FC<{
  manifest: GymEnvironmentManifest;
  fileCount: number;
  totalSize: number;
}> = ({ manifest, fileCount, totalSize }) => (
  <Block className="bg-surface-sunken rounded-lg" padding="density-lg">
    <Stack gap="density-md">
      <Text kind="label/bold/sm">Environment</Text>
      <Stack gap="density-sm">
        {manifest.envName && (
          <KVPair orientation="horizontal" label="Name" value={manifest.envName} />
        )}
        <KVPair orientation="horizontal" label="Format" value={manifest.format} />
        {manifest.description && (
          <KVPair orientation="horizontal" label="Description" value={manifest.description} />
        )}
        {manifest.hubId && (
          <KVPair orientation="horizontal" label="Source" value={manifest.hubId} />
        )}
        {manifest.vfEnvId && (
          <KVPair orientation="horizontal" label="Env ID" value={manifest.vfEnvId} />
        )}
        <KVPair
          orientation="horizontal"
          label="Contents"
          value={`${fileCount} ${fileCount === 1 ? 'file' : 'files'}${manifest.wheelCount > 0 ? ` · ${manifest.wheelCount} ${manifest.wheelCount === 1 ? 'wheel' : 'wheels'}` : ''} · ${getHumanReadableFileSize(totalSize)}`}
        />
      </Stack>
    </Stack>
  </Block>
);

export const RewardEnvironmentSection: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { control } = useFormContext<CustomizationFormFields>();

  const {
    field: { value: selectedRef, onChange: setSelectedRef, onBlur },
    fieldState: { error: fieldError },
  } = useController({ control, name: 'grpo.environmentFileset' });

  // `nemo` uploads gym packages with purpose=environment, and the backend checks it at
  // submit. It also still accepts `generic`, but the filter takes one value, so listing
  // environment is what keeps a properly uploaded package selectable.
  const { data: filesetsResponse, isPending } = useFilesListFilesets(workspace, {
    page_size: 100,
    sort: '-updated_at',
    filter: { purpose: FilesetPurpose.environment },
  });
  const filesets = useMemo(() => filesetsResponse?.data ?? [], [filesetsResponse?.data]);

  const selectedParts = selectedRef ? getPartsFromReference(selectedRef as string) : undefined;
  const filesetName = selectedParts?.name ?? '';

  const {
    manifest,
    isPending: isManifestPending,
    fileCount,
    totalSize,
    noConfigWarning,
    manifestIssues,
  } = useGymEnvironmentManifest({ workspace, filesetName });

  const handleSelectChange = (value: string) => {
    const picked = filesets.find((f) => getEntityReference(f) === value);
    if (picked) setSelectedRef(getEntityReference(picked));
  };

  const hasSelection = !!(selectedRef as string);
  const noEnvs = !isPending && filesets.length === 0;

  return (
    <FormSection
      title="Reward Environment"
      description={
        <>
          GRPO scores sampled responses using a NeMo Gym environment. Upload one to a fileset first,
          see{' '}
          <Anchor href={LINK_DOCS_GRPO_TRAINING} target="_blank" rel="noopener noreferrer">
            GRPO Environment Guide
          </Anchor>
          .
        </>
      }
    >
      <Stack gap="density-lg">
        <FormField
          slotLabel="Environment"
          slotInfo="NeMo Gym package that runs the rollouts and scores them. Filesets uploaded with purpose 'environment' appear here."
          slotError={fieldError?.message}
          status={fieldError ? 'error' : undefined}
        >
          <SelectRoot
            value={(selectedRef as string) ?? ''}
            onValueChange={handleSelectChange}
            onOpenChange={(open) => {
              if (!open) onBlur();
            }}
            disabled={isPending}
          >
            <SelectTrigger
              aria-label="environment-fileset-select"
              placeholder={isPending ? 'Loading…' : 'Select a reward environment fileset'}
            />
            <SelectContent>
              <SelectListbox>
                {filesets.map((f) => {
                  const ref = getEntityReference(f);
                  return (
                    <SelectItem key={ref} value={ref}>
                      {f.name}
                    </SelectItem>
                  );
                })}
                {noEnvs && (
                  <Block paddingX="density-md" paddingY="density-sm">
                    <Text className="text-secondary" kind="body/regular/sm">
                      No environment filesets found.
                    </Text>
                  </Block>
                )}
              </SelectListbox>
            </SelectContent>
          </SelectRoot>
        </FormField>

        {hasSelection && !isManifestPending && (
          <Stack gap="density-sm">
            {noConfigWarning && (
              <Banner kind="inline" status="warning">
                No <code>nemo-environment.yaml</code> found. This fileset is not a valid NeMo Gym
                environment package and will be rejected at training time.
              </Banner>
            )}

            {manifestIssues.length > 0 && (
              <Banner kind="inline" status="warning">
                <Stack gap="density-xs">
                  <Text kind="body/regular/sm">
                    This environment package will be rejected at training time:
                  </Text>
                  {manifestIssues.map((issue) => (
                    <Text key={issue} kind="body/regular/sm">
                      • {issue}
                    </Text>
                  ))}
                </Stack>
              </Banner>
            )}

            {manifest && (
              <GymEnvironmentInfo manifest={manifest} fileCount={fileCount} totalSize={totalSize} />
            )}
          </Stack>
        )}
      </Stack>
    </FormSection>
  );
};
