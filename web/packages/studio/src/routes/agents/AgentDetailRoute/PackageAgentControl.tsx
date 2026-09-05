// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { FormModal } from '@nemo/common/src/components/FormModal';
import {
  Accordion,
  Badge,
  Button,
  CodeSnippetActions,
  CodeSnippetCode,
  CodeSnippetRoot,
  Flex,
  FormField,
  Spinner,
  Stack,
  StatusIndicator,
  Text,
  TextInput,
} from '@nvidia/foundations-react-core';
import { usePackageAgent } from '@studio/api/agents/usePackageAgent';
import { CopyButton } from '@studio/components/CopyButton';
import { JOBS_ENABLED } from '@studio/constants/environment';
import { getWorkspaceJobDetailRoute } from '@studio/routes/utils';
import { Package } from 'lucide-react';
import { useEffect, useState, type FC } from 'react';
import { useNavigate } from 'react-router';

interface PackageAgentControlProps {
  workspace: string;
  agentName: string;
  /** Packaging needs a Platform-managed agent config, the same gate deploying uses. */
  canPackage: boolean;
  /** Offers the finished tag to the deployment flow. */
  onImageBuilt?: (image: string) => void;
  /**
   * Reports the finished tag as soon as the build produces one, so deploying
   * from anywhere on the page starts from the image this agent just built.
   */
  onImageAvailable?: (image: string) => void;
}

/**
 * Build a container image for this agent so a docker or k8s deployment has one.
 *
 * Packaging is a precondition of one deployment mode rather than a peer of the
 * deployments list, so it lives behind this button instead of a panel above it.
 * The build runs for minutes, so the trigger reports progress while the modal is
 * closed — otherwise starting a build and navigating away would lose it.
 */
export const PackageAgentControl: FC<PackageAgentControlProps> = ({
  workspace,
  agentName,
  canPackage,
  onImageBuilt,
  onImageAvailable,
}) => {
  const navigate = useNavigate();
  const [isOpen, setIsOpen] = useState(false);
  const [registry, setRegistry] = useState('');
  const [pushOptionsOpen, setPushOptionsOpen] = useState<string | undefined>();
  const {
    packageAgent,
    submitError,
    isSubmitting,
    jobName,
    isRunning,
    isQueued,
    isStalled,
    isComplete,
    isFailed,
    isRestored,
    image,
    published,
  } = usePackageAgent({ workspace, agentName });

  useEffect(() => {
    if (isComplete && image) {
      onImageAvailable?.(image);
    }
  }, [isComplete, image, onImageAvailable]);

  const isBusy = isQueued || isRunning;
  const hasImage = isComplete && Boolean(image);

  return (
    <>
      <Flex gap="density-sm" align="center">
        {/* Status rides beside the button, not inside it — a noun and a dot on the
            control itself read as a badge rather than something to press. */}
        {hasImage && !isBusy ? (
          <Badge color="green" kind="outline" size="small">
            Image ready
          </Badge>
        ) : null}
        <Button kind="secondary" size="small" onClick={() => setIsOpen(true)}>
          {isBusy ? (
            <Spinner size="small" aria-label="Build in progress" />
          ) : (
            <Package className="size-4" aria-hidden />
          )}
          {isBusy ? 'Building…' : hasImage ? 'Manage image' : 'Build image'}
        </Button>
      </Flex>

      <FormModal
        open={isOpen}
        title="Container image"
        instruction="Build an image for this agent to deploy it with Docker or Kubernetes."
        submitButtonText={hasImage ? 'Rebuild' : 'Build image'}
        cancelButtonText="Close"
        loading={isSubmitting}
        submitDisabled={!canPackage || isRunning}
        onClose={() => setIsOpen(false)}
        onSubmit={(e) => {
          e.preventDefault();
          packageAgent(registry.trim() ? { registry: registry.trim() } : {});
        }}
      >
        {!canPackage ? (
          <Text kind="body/regular/sm" color="secondary">
            Packaging is available for Platform-managed agents. Build a NAT workflow image with{' '}
            <code>nemo agents package</code>.
          </Text>
        ) : null}

        {submitError ? (
          <Text kind="body/regular/sm" color="danger">
            {getErrorMessage(submitError as Error, 'Failed to start the packaging job')}
          </Text>
        ) : null}

        {hasImage && image ? (
          <Stack gap="density-sm">
            <CodeSnippetRoot className="[&_.nv-code-snippet-actions]:justify-between">
              <CodeSnippetActions>
                <Flex gap="density-sm" align="center">
                  <StatusIndicator color="green" size="small" />
                  <Text kind="label/bold/sm">
                    {isRestored ? 'Image ready' : 'Build finished — image ready'}
                  </Text>
                </Flex>
                <CopyButton text={image} color="neutral" kind="tertiary" size="tiny" />
              </CodeSnippetActions>
              <CodeSnippetCode value={image} />
            </CodeSnippetRoot>
            {published ? (
              <Text kind="body/regular/sm" color="secondary">
                Pushed to {published}
              </Text>
            ) : null}
          </Stack>
        ) : null}

        {isComplete && !image ? (
          <Text kind="body/regular/sm" color="secondary">
            The job finished without reporting an image tag. Open the job to see why.
          </Text>
        ) : null}

        {isStalled ? (
          <Text kind="body/regular/sm" className="text-warning">
            The job was accepted but has not started. Check that the platform is running a jobs
            controller.
          </Text>
        ) : null}

        {isFailed ? (
          <Text kind="body/regular/sm" color="danger">
            Packaging failed. Open the job for the build output.
          </Text>
        ) : null}

        {isBusy && !isStalled ? (
          <Flex gap="density-sm" className="items-center">
            <Spinner size="small" aria-label="Building image" />
            <Text kind="body/regular/sm" color="secondary">
              {isQueued ? 'Waiting for a build to start…' : 'Building — this takes a few minutes.'}
            </Text>
          </Flex>
        ) : null}

        {/* Build inputs belong to the *next* build, so they are absent once an image
            exists — sitting above a finished tag they read as describing it. */}
        {canPackage && !isBusy && !hasImage ? (
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

        {(hasImage && image && onImageBuilt) || (jobName && JOBS_ENABLED) ? (
          <Flex gap="density-sm" align="center">
            {hasImage && image && onImageBuilt ? (
              <Button
                kind="primary"
                size="small"
                type="button"
                onClick={() => {
                  setIsOpen(false);
                  onImageBuilt(image);
                }}
              >
                Deploy
              </Button>
            ) : null}
            {jobName && JOBS_ENABLED ? (
              <Button
                kind="tertiary"
                size="small"
                type="button"
                onClick={() => navigate(getWorkspaceJobDetailRoute(workspace, jobName))}
              >
                View job
              </Button>
            ) : null}
          </Flex>
        ) : null}
      </FormModal>
    </>
  );
};
