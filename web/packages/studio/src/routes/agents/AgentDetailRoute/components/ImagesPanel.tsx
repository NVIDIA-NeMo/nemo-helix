// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { formatFileSize } from '@nemo/common/src/components/UploadModal/utils';
import {
  getAgentsListAgentImagesQueryKey,
  useAgentsDeleteAgentImage,
  useAgentsListAgentImages,
} from '@nemo/sdk/generated/agents/agent-images';
import type { AgentImage } from '@nemo/sdk/generated/agents/schema/AgentImage';
import { Badge, Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { AgentSpecSource } from '@studio/api/agents/useAgentSpecFileset';
import { enabledImageModes, useDeploymentModes } from '@studio/api/agents/useDeploymentModes';
import { usePackageAgent } from '@studio/api/agents/usePackageAgent';
import { StackedSkeleton } from '@studio/components/StackedSkeleton';
import { CommitLink } from '@studio/routes/agents/AgentDetailRoute/components/CommitLink';
import { ImageBuildRow } from '@studio/routes/agents/AgentDetailRoute/components/ImageBuildRow';
import { DetailPanel } from '@studio/routes/agents/AgentDetailRoute/overview/DetailPanel';
import { PackageAgentControl } from '@studio/routes/agents/AgentDetailRoute/PackageAgentControl';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useEffect, useState } from 'react';

interface ImagesPanelProps {
  workspace: string;
  agentName: string;
  /** Packaging is Fabric-only. */
  canPackage: boolean;
  /** Still resolving whether this agent can be packaged, which is not the same as "no". */
  isAgentLoading?: boolean;
  /** Where the agent's files come from, to link the commit each image was built from. */
  specSource?: AgentSpecSource;
  /** Deploy this image tag. */
  onDeployImage?: (image: string) => void;
  /** Reports a tag built on this page load, so deploying elsewhere starts from it. */
  onImageAvailable?: (image: string) => void;
}

const builtAt = (image: AgentImage) => (image.created ? Date.parse(image.created) || 0 : 0);

interface ImageRowProps {
  image: AgentImage;
  isFirst: boolean;
  isLatest: boolean;
  specSource?: AgentSpecSource;
  onDeploy?: () => void;
  onRemove: () => void;
}

const ImageRow: FC<ImageRowProps> = ({
  image,
  isFirst,
  isLatest,
  specSource,
  onDeploy,
  onRemove,
}) => (
  <Flex
    align="center"
    gap="2"
    data-testid="image-row"
    className={`px-4 py-3 ${isFirst ? '' : 'border-t border-base'}`}
  >
    <Stack gap="0" className="min-w-0 flex-1">
      {image.tags.map((tag) => (
        <Text key={tag} kind="body/semibold/sm" className="truncate font-mono" title={tag}>
          {tag}
        </Text>
      ))}
      <Text kind="body/regular/xs" className="truncate text-secondary">
        {[
          image.size ? formatFileSize(image.size) : null,
          image.created ? `Built ${new Date(image.created).toLocaleString()}` : null,
        ]
          .filter(Boolean)
          .join(' · ')}
        {image.spec_revision ? (
          <>
            {' from commit '}
            <CommitLink source={specSource} revision={image.spec_revision} />
          </>
        ) : null}
        {image.deployments?.length ? ` · Used by ${image.deployments.join(', ')}` : null}
      </Text>
    </Stack>
    <Flex align="center" gap="2" className="shrink-0">
      {isLatest ? (
        <Badge kind="outline" color="gray" size="small">
          Latest
        </Badge>
      ) : null}
      <Badge kind="solid" color={image.in_use ? 'green' : 'gray'}>
        {image.in_use ? 'In use' : 'Unused'}
      </Badge>
      <Flex gap="1">
        {onDeploy ? (
          <Button kind="tertiary" size="small" onClick={onDeploy}>
            Deploy
          </Button>
        ) : null}
        {/* Removing an in-use image would break the deployments running it. */}
        <Button
          kind="tertiary"
          size="small"
          color="danger"
          disabled={image.in_use}
          onClick={onRemove}
        >
          Remove
        </Button>
      </Flex>
    </Flex>
  </Flex>
);

/** Builds and lists this agent's images, newest first; each can be deployed or, if unused, removed. */
export const ImagesPanel: FC<ImagesPanelProps> = ({
  workspace,
  agentName,
  canPackage,
  isAgentLoading,
  specSource,
  onDeployImage,
  onImageAvailable,
}) => {
  const queryClient = useQueryClient();
  const params = { agent: agentName };
  const queryKey = getAgentsListAgentImagesQueryKey(workspace, params);
  const { data, isLoading } = useAgentsListAgentImages(workspace, params);
  const deleteImage = useAgentsDeleteAgentImage();
  const build = usePackageAgent({ workspace, agentName });
  const canDeployImages = enabledImageModes(useDeploymentModes(workspace)).length > 0;
  const [target, setTarget] = useState<AgentImage | null>(null);
  // Set while the list refetches for a build that just finished, so the new image
  // has a placeholder instead of appearing out of nowhere.
  const [isFetchingBuilt, setIsFetchingBuilt] = useState(false);
  const images = [...(data?.data ?? [])].sort((a, b) => builtAt(b) - builtAt(a));

  const { isComplete, isRestored, image: builtImage } = build;
  useEffect(() => {
    // Only a build watched on this page load. A restored tag can be months old
    // and from an earlier revision, so it must not silently become the default
    // image for every deployment.
    if (!isComplete || !builtImage || isRestored) return;
    setIsFetchingBuilt(true);
    void queryClient
      .invalidateQueries({
        queryKey: getAgentsListAgentImagesQueryKey(workspace, { agent: agentName }),
      })
      .finally(() => setIsFetchingBuilt(false));
    onImageAvailable?.(builtImage);
  }, [isComplete, builtImage, isRestored, onImageAvailable, queryClient, workspace, agentName]);

  const body = () => {
    if (isLoading || isFetchingBuilt) {
      return (
        <Stack gap="2" className="p-4" data-testid="images-loading">
          <StackedSkeleton count={Math.max(images.length + 1, 2)} height={40} className="w-full" />
        </Stack>
      );
    }
    if (images.length === 0) {
      return (
        <Text kind="body/regular/sm" className="p-4 text-secondary">
          No images built for this agent.
        </Text>
      );
    }
    return images.map((image, index) => {
      const tag = image.tags[0];
      return (
        <ImageRow
          key={image.id}
          image={image}
          isFirst={index === 0}
          isLatest={index === 0 && images.length > 1}
          specSource={specSource}
          onDeploy={onDeployImage && canDeployImages && tag ? () => onDeployImage(tag) : undefined}
          onRemove={() => setTarget(image)}
        />
      );
    });
  };

  const buildRow = <ImageBuildRow workspace={workspace} build={build} />;

  return (
    <>
      <DetailPanel
        title="Images"
        flush
        slotAction={
          <PackageAgentControl
            workspace={workspace}
            agentName={agentName}
            build={build}
            canPackage={canPackage}
            isAgentLoading={isAgentLoading}
          />
        }
      >
        <Stack gap="0" className="[&>*+*]:border-t [&>*+*]:border-base">
          {buildRow}
          <Stack gap="0">{body()}</Stack>
        </Stack>
      </DetailPanel>
      {target && (
        <DeleteConfirmationModal
          open
          title="Remove Image"
          description={`Remove ${target.tags.join(', ')} from the deployment host? Deploying it again requires a new build.`}
          successText="Image removed."
          onDelete={async () => {
            await deleteImage.mutateAsync({ workspace, imageId: target.id });
            await queryClient.invalidateQueries({ queryKey });
            return true;
          }}
          onClose={() => setTarget(null)}
        />
      )}
    </>
  );
};
