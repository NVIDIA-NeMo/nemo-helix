// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useStageVictimContext } from '@agent-hardener/components/useStageVictimContext';
import {
  Banner,
  Button,
  Flex,
  FormField,
  Panel,
  Stack,
  Text,
  TextInput,
} from '@nvidia/foundations-react-core';
import { FC, useEffect, useState } from 'react';

interface VictimImagePanelProps {
  workspace: string;
  manifestName: string;
  /** The image recorded on the manifest; empty when none is set. */
  victimImage: string;
  saving: boolean;
  onSave: (victimImage: string) => void;
}

const IMAGE_PLACEHOLDER = '<registry>/<image>:<tag>';

/** Image reference syntax only — the registry is never contacted from here. Empty clears the image. */
const isValidImageRef = (value: string): boolean => !/\s/.test(value);

/**
 * The victim image a containerized war-game runs, and the steps to produce it.
 *
 * Where war-games run in a container (NeMo Helix on Kubernetes) the job cannot build the victim, and
 * OpenShell creates sandboxes only from an image the cluster can pull. So the operator stages the build
 * context here, builds and pushes it, and records the pushed reference. Runs stay blocked until then.
 */
export const VictimImagePanel: FC<VictimImagePanelProps> = ({
  workspace,
  manifestName,
  victimImage,
  saving,
  onSave,
}) => {
  const stage = useStageVictimContext(workspace, manifestName);
  const [draft, setDraft] = useState(victimImage);
  useEffect(() => setDraft(victimImage), [victimImage]);
  const valid = isValidImageRef(draft.trim());

  const image = draft.trim() || IMAGE_PLACEHOLDER;
  const commands = [`docker build -t ${image} - < ${stage.archiveName}`, `docker push ${image}`].join('\n');

  return (
    <Panel>
      <Stack gap="density-lg" padding="density-lg">
        <Text kind="body/semibold/md">Victim Image</Text>
        <Text kind="body/regular/sm" className="text-fg-secondary">
          War-games here run on Kubernetes, which cannot build the victim. Stage its build context, build
          and push the image to a registry the OpenShell gateway can pull from, then record it below.
          Re-stage after refreshing the target or upgrading Agent Hardener: a run rejects an image built
          from a different context.
        </Text>

        {victimImage ? null : (
          <Banner kind="inline" status="warning">
            <Text kind="body/regular/sm">
              No victim image yet. War-games and benign-suite generation stay disabled until one is set.
            </Text>
          </Banner>
        )}

        <Flex gap="density-sm" className="items-center">
          <Button kind="secondary" size="small" disabled={stage.staging} onClick={stage.start}>
            {stage.staging ? 'Staging…' : 'Stage Build Context'}
          </Button>
          {stage.ready ? (
            <Button
              kind="secondary"
              size="small"
              disabled={stage.downloading}
              onClick={() => void stage.download()}
            >
              {stage.downloading ? 'Downloading…' : `Download ${stage.archiveName}`}
            </Button>
          ) : null}
        </Flex>
        {stage.failedJob ? (
          <Text kind="body/regular/sm" className="text-fg-error">
            Staging failed. See the logs of job {stage.failedJob}.
          </Text>
        ) : null}

        <pre className="overflow-auto rounded bg-surface-sunken p-2 text-xs text-primary">{commands}</pre>

        <Flex gap="density-sm" className="items-end">
          <FormField
            className="grow"
            name="victim_image"
            slotLabel="Image Reference"
            slotError={valid ? undefined : 'An image reference cannot contain whitespace.'}
          >
            <TextInput
              value={draft}
              placeholder={IMAGE_PLACEHOLDER}
              onChange={(event) => setDraft(event.target.value)}
            />
          </FormField>
          <Button
            kind="secondary"
            disabled={saving || !valid || draft.trim() === victimImage}
            onClick={() => onSave(draft.trim())}
          >
            {saving ? 'Saving…' : 'Save'}
          </Button>
        </Flex>
      </Stack>
    </Panel>
  );
};
