// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { Package } from '@nemo/icons/icons';
import {
  Accordion,
  Button,
  CodeSnippetActions,
  CodeSnippetCode,
  CodeSnippetRoot,
  FormField,
  Stack,
  Text,
  TextInput,
} from '@nvidia/foundations-react-core';
import {
  type DeploymentMode,
  enabledImageModes,
  IMAGE_DEPLOYMENT_MODES,
  useDeploymentModes,
} from '@studio/api/agents/useDeploymentModes';
import { useImageBuildsUnsupported } from '@studio/api/agents/useImageBuildsUnsupported';
import type { PackageAgentBuild } from '@studio/api/agents/usePackageAgent';
import { CopyButton } from '@studio/components/CopyButton';
import { deploymentModeLabel } from '@studio/routes/agents/AgentDetailRoute/helpers';
import { useState, type FC } from 'react';

const modeLabels = (modes: readonly DeploymentMode[]) =>
  modes.map(deploymentModeLabel).join(' or ');

const CommandSnippet: FC<{ command: string }> = ({ command }) => (
  <CodeSnippetRoot>
    <CodeSnippetActions>
      <CopyButton text={command} color="neutral" kind="tertiary" size="tiny" />
    </CodeSnippetActions>
    <CodeSnippetCode value={command} language="bash" />
  </CodeSnippetRoot>
);

// `mode` is undefined when the platform's modes couldn't be read.
const LocalBuildSteps: FC<{ workspace: string; agentName: string; mode?: DeploymentMode }> = ({
  workspace,
  agentName,
  mode,
}) => {
  const publishes = mode !== 'docker';
  return (
    <Stack gap="density-sm">
      <Text className="text-secondary" kind="body/regular/sm">
        This platform can&apos;t build images: builds run on the platform host under a subprocess
        job profile, and this platform doesn&apos;t register one. Build the image yourself, from the
        agent&apos;s directory{publishes ? ', replacing REGISTRY with your registry' : ''}:
      </Text>
      <CommandSnippet
        command={`nemo agents package --agent agent.yaml${publishes ? ' --publish --registry REGISTRY' : ''}`}
      />
      <Text className="text-secondary" kind="body/regular/sm">
        Then deploy it, replacing IMAGE with the tag the build printed
        {mode ? '' : ' and MODE with docker or k8s'}:
      </Text>
      <CommandSnippet
        command={`nemo agents deploy --agent ${agentName} --workspace ${workspace} --mode ${mode ?? 'MODE'} --image IMAGE`}
      />
    </Stack>
  );
};

interface PackageAgentControlProps {
  workspace: string;
  agentName: string;
  /** The agent's build, owned by the images panel so its list can show progress. */
  build: PackageAgentBuild;
  /** Packaging needs a Platform-managed agent config, the same gate deploying uses. */
  canPackage: boolean;
  /** Still resolving whether this agent can be packaged, which is not the same as "no". */
  isAgentLoading?: boolean;
}

/**
 * Start a container image build for this agent. The modal only starts the build:
 * progress, failures, and the finished image show in the images list, which
 * outlives the modal.
 */
export const PackageAgentControl: FC<PackageAgentControlProps> = ({
  workspace,
  agentName,
  build,
  canPackage,
  isAgentLoading = false,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [registry, setRegistry] = useState('');
  const [pushOptionsOpen, setPushOptionsOpen] = useState<string | undefined>();
  const { packageAgent, submitError, isSubmitting, isRunning, isQueued } = build;
  const platformCannotBuild = useImageBuildsUnsupported();
  const deploymentModes = useDeploymentModes(workspace);
  const imageModes = enabledImageModes(deploymentModes);
  const noImageMode = imageModes.length === 0;
  const knownDeployMode: DeploymentMode | undefined =
    deploymentModes.status === 'ready' ? imageModes.at(0) : undefined;
  const isBusy = isQueued || isRunning;

  const showLocalBuild =
    canPackage && platformCannotBuild && !noImageMode && deploymentModes.status !== 'loading';

  return (
    <>
      {/* Progress shows in the images list; the trigger only says a build can't start now. */}
      <Button kind="secondary" size="small" disabled={isBusy} onClick={() => setIsOpen(true)}>
        <Package className="size-4" aria-hidden />
        Build image
      </Button>

      <FormModal
        open={isOpen}
        title="Build Image"
        instruction={`Build an image from this agent's current source to deploy it with ${modeLabels(
          noImageMode ? IMAGE_DEPLOYMENT_MODES : imageModes
        )}.`}
        submitButtonText="Build image"
        cancelButtonText="Close"
        loading={isSubmitting}
        submitDisabled={!canPackage || isBusy || platformCannotBuild || noImageMode}
        onClose={() => setIsOpen(false)}
        onSubmit={(e) => {
          e.preventDefault();
          packageAgent(registry.trim() ? { registry: registry.trim() } : {}, {
            onSuccess: () => setIsOpen(false),
          });
        }}
      >
        {isAgentLoading ? (
          <Text className="text-secondary" kind="body/regular/sm">
            Checking whether this agent can be packaged…
          </Text>
        ) : null}

        {!canPackage && !isAgentLoading ? (
          <Text className="text-secondary" kind="body/regular/sm">
            Packaging is available for Platform-managed agents. Build a NAT workflow image with{' '}
            <code>nemo agents package</code>.
          </Text>
        ) : null}

        {canPackage && noImageMode ? (
          <Text className="text-secondary" kind="body/regular/sm">
            This platform has no Docker or Kubernetes deployment mode enabled, so an image built
            here couldn&apos;t be deployed. Ask your platform admin to configure a Docker or
            Kubernetes executor.
          </Text>
        ) : null}

        {showLocalBuild ? (
          <LocalBuildSteps workspace={workspace} agentName={agentName} mode={knownDeployMode} />
        ) : null}

        {submitError ? (
          <Text kind="body/regular/sm" className="text-feedback-danger">
            {getErrorMessage(submitError, 'Failed to start the packaging job')}
          </Text>
        ) : null}

        {canPackage && !isBusy && !platformCannotBuild && !noImageMode ? (
          <Accordion
            className="[&>div]:border-b-0"
            value={pushOptionsOpen}
            onValueChange={setPushOptionsOpen}
            items={[
              {
                value: 'push',
                chevronPosition: 'start',
                slotTrigger: 'Push options',
                slotContent: (
                  <FormField
                    className="pt-density-sm"
                    slotLabel="Registry (optional)"
                    slotHelp="Push the built image here so a cluster can pull it. The platform host must already be logged in to it — credentials are never sent through this form."
                  >
                    <TextInput
                      value={registry}
                      placeholder="nvcr.io/my-org"
                      onValueChange={setRegistry}
                    />
                  </FormField>
                ),
              },
            ]}
          />
        ) : null}
      </FormModal>
    </>
  );
};
