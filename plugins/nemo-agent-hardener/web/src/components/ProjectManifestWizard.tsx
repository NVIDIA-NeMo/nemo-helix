// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { useUploadProjectFileset } from '@agent-hardener/api/filesets';
import { ModelGroupFields } from '@agent-hardener/components/ModelGroupFields';
import { parseEnvPairs, splitList } from '@agent-hardener/formValues';
import {
  useAgentHardenerGetModelConfigDefaults,
  useAgentHardenerInspectProject,
} from '@agent-hardener/generated/api';
import type { InspectProjectResponse, WarGameModels } from '@agent-hardener/generated/schema';
import { useToast } from '@agent-hardener/host';
import { AccordionSection, ControlledSelect, ControlledTextInput, FileUpload } from '@nemo/common';
import { AccordionRoot, Banner, Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { FC, useState } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';

/** The project-source fields the wizard hands back to the create-manifest call. */
export interface ProjectManifestValues {
  project_fileset: string;
  workflow: string;
  launch_mode: string;
  port: number;
  secrets: string[];
  secrets_file: string;
  egress: string[];
  backends: string[];
  env: Record<string, string>;
  models: WarGameModels;
}

interface ProjectManifestWizardProps {
  workspace: string;
  /** Manifest id collected up front (used to name the fileset and gate detection). */
  manifestName: string;
  /** Whether the manifest id passes validation — detection is blocked until it does. */
  nameValid: boolean;
  isCreating: boolean;
  onCreate: (values: ProjectManifestValues) => void;
}

/**
 * Upload a NAT project, detect its layout (`agent-hardener inspect`), then confirm the detected defaults.
 *
 * A web clone of `agent-hardener init`: the project comes first, so every substantive answer (workflow,
 * secrets, port) is seeded from detection rather than asked blind.
 */
export const ProjectManifestWizard: FC<ProjectManifestWizardProps> = ({
  workspace,
  manifestName,
  nameValid,
  isCreating,
  onCreate,
}) => {
  const toast = useToast();
  const [file, setFile] = useState<File | undefined>();
  const [detection, setDetection] = useState<InspectProjectResponse | undefined>();
  const [filesetRef, setFilesetRef] = useState<string | undefined>();

  const uploadProject = useUploadProjectFileset();
  const inspectProject = useAgentHardenerInspectProject();
  const detecting = uploadProject.isPending || inspectProject.isPending;

  const detect = async () => {
    if (!file) return;
    try {
      const ref = await uploadProject.mutateAsync({ workspace, manifestName, file });
      const detected = await inspectProject.mutateAsync({
        workspace,
        data: { project_fileset: ref },
      });
      setFilesetRef(ref);
      setDetection(detected);
    } catch {
      toast.error(
        'Could not inspect the uploaded project. Check that it is a valid NAT project zip.'
      );
    }
  };

  if (detection && filesetRef) {
    return (
      <ProjectReviewForm
        detection={detection}
        filesetRef={filesetRef}
        workspace={workspace}
        isCreating={isCreating}
        onCreate={onCreate}
        onReset={() => {
          setDetection(undefined);
          setFilesetRef(undefined);
        }}
      />
    );
  }

  return (
    <Stack gap="density-lg">
      <Text kind="body/regular/md" className="text-subtle">
        Upload your NAT project as a single zip (workflow plus its tool code). We inspect it to
        detect the workflow, secrets, and egress — nothing is executed.
      </Text>
      <FileUpload
        label="Project Archive"
        accept={{ 'application/zip': ['.zip'] }}
        multiple={false}
        files={file ? [file] : []}
        onDropAccepted={(accepted) => setFile(accepted[0])}
        onRemoveFile={() => setFile(undefined)}
        helperText="A single .zip containing an installable NAT project (pyproject.toml + workflow)."
      />
      <Flex gap="density-md">
        <Button color="brand" onClick={detect} disabled={!file || !nameValid || detecting}>
          {detecting ? 'Detecting…' : 'Detect Project'}
        </Button>
        {!nameValid && (
          <Text kind="body/regular/sm" className="self-center text-subtle">
            Enter a valid manifest ID above first.
          </Text>
        )}
      </Flex>
    </Stack>
  );
};

const reviewSchema = z.object({
  workflow: z.string().trim().min(1, 'Select a workflow'),
  port: z.coerce.number().int().positive('Enter a valid port'),
  secrets: z.string().trim(),
  secretsFile: z.string().trim(),
  egress: z.string().trim(),
  backends: z.string().trim(),
  env: z.string().trim(),
});
type ReviewData = z.infer<typeof reviewSchema>;

interface ProjectReviewFormProps {
  detection: InspectProjectResponse;
  filesetRef: string;
  workspace: string;
  isCreating: boolean;
  onCreate: (values: ProjectManifestValues) => void;
  onReset: () => void;
}

const ProjectReviewForm: FC<ProjectReviewFormProps> = ({
  detection,
  filesetRef,
  workspace,
  isCreating,
  onCreate,
  onReset,
}) => {
  const workflows = detection.workflows ?? [];
  const [models, setModels] = useState<WarGameModels>({});
  const { data: modelDefaults } = useAgentHardenerGetModelConfigDefaults(workspace, { query: {} });

  const { control, handleSubmit } = useForm<ReviewData>({
    resolver: zodResolver(reviewSchema),
    defaultValues: {
      workflow: workflows[0] ?? '',
      port: detection.default_port ?? 8000,
      secrets: (detection.secret_names ?? []).join(', '),
      secretsFile: detection.secrets_file ?? '',
      egress: (detection.egress ?? []).join(', '),
      backends: (detection.backend_ports ?? []).map((port) => `backend-${port}:${port}`).join(', '),
      // Nothing to seed: `agent-hardener inspect` reports no env, so this is the operator's to fill.
      env: '',
    },
  });

  const onSubmit = handleSubmit((data) =>
    onCreate({
      project_fileset: filesetRef,
      workflow: data.workflow,
      launch_mode: 'workflow',
      port: data.port,
      secrets: splitList(data.secrets),
      secrets_file: data.secretsFile,
      egress: splitList(data.egress),
      backends: splitList(data.backends),
      env: parseEnvPairs(data.env),
      models,
    })
  );

  return (
    <form onSubmit={onSubmit}>
      <Stack gap="density-lg">
        <Banner status="info" kind="inline">
          Project detected. Confirm the settings below, then create.
        </Banner>
        <ControlledSelect
          useControllerProps={{ control, name: 'workflow' }}
          items={workflows.map((wf) => ({ value: wf, children: wf }))}
          formFieldProps={{
            slotLabel: 'Workflow',
            slotHelp: 'The workflow file the victim serves.',
          }}
        />
        <ControlledTextInput
          useControllerProps={{ control, name: 'port' }}
          formFieldProps={{ slotLabel: 'Victim Port' }}
        />
        <ControlledTextInput
          useControllerProps={{ control, name: 'secrets' }}
          formFieldProps={{
            slotLabel: 'Secret Names',
            slotHelp: 'Comma-separated; values come from the operator env.',
          }}
        />
        <ControlledTextInput
          useControllerProps={{ control, name: 'secretsFile' }}
          formFieldProps={{
            slotLabel: 'Secrets File (optional)',
            slotHelp: 'Dotenv path within the project.',
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
        <ControlledTextInput
          useControllerProps={{ control, name: 'backends' }}
          formFieldProps={{
            slotLabel: 'Host Backends',
            slotHelp:
              'Comma-separated NAME:PORT for host services the tools call on localhost (a DB/API). ' +
              'Agent Hardener rewrites localhost:PORT to your host and opens the route. Detected ports are prefilled.',
          }}
        />
        <ControlledTextInput
          useControllerProps={{ control, name: 'egress' }}
          formFieldProps={{
            slotLabel: 'Egress Allow-list',
            slotHelp:
              'Comma-separated host[:port] for external services the agent calls (e.g. inference-api.nvidia.com).',
          }}
        />
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
        <Flex gap="density-md">
          <Button color="brand" type="submit" disabled={isCreating}>
            {isCreating ? 'Creating…' : 'Create Manifest'}
          </Button>
          <Button kind="tertiary" type="button" onClick={onReset}>
            Upload a Different Project
          </Button>
        </Flex>
      </Stack>
    </form>
  );
};
