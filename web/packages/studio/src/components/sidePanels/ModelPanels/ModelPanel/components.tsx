// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ExpandableMessage } from '@nemo/common/src/components/ExpandableMessage';
import { KVPair } from '@nemo/common/src/components/KVPair';
import { parametersToString } from '@nemo/common/src/components/ModelDetailsTooltip/utils';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { getModelVersion, TagLabel, tagToIcon } from '@nemo/common/src/constants/modelMetadata';
import { kebabCaseToTitleCase } from '@nemo/common/src/utils/formatters';
import { isDefined } from '@nemo/common/src/utils/list';
import {
  ModelDeploymentStatus,
  type Adapter,
  type ModelEntity,
  type ModelDeployment,
} from '@nemo/sdk/generated/platform/schema';
import {
  Accordion,
  Anchor,
  Badge,
  Button,
  CodeSnippetActions,
  CodeSnippetCode,
  CodeSnippetRoot,
  Divider,
  Flex,
  Stack,
  StatusIndicator,
  Text,
  Tooltip,
} from '@nvidia/foundations-react-core';
import {
  ChatMulti,
  Eye,
  EyeOff,
  Layers,
  HAlignLeft,
  IconWindow,
} from '@nvidia/react-gui-icons/icons';
import { CopyButton } from '@studio/components/CopyButton';
import { getWorkspaceCustomizationJobDetailsRoute } from '@studio/routes/utils';
import { formatFinetuningType } from '@studio/util/customizations';
import { FC, useState } from 'react';

export type ModelParameters = ModelDetailOverviewProps & ModelParametersAccordionProps;

export interface ModelDetailOverviewProps {
  model: ModelEntity;
  /** Optional description override (defaults to model.description) */
  description?: string;
  /** Badges for the model */
  badges?: TagLabel[];
  /** Optional deployment status when available from deployment service */
  status?: ModelDeploymentStatus;
  slotActions?: React.ReactNode;
}

export const ModelDetailOverview: FC<ModelDetailOverviewProps> = ({
  model,
  description: descriptionOverride,
  badges,
  status,
  slotActions,
}) => {
  const spec = model.spec;
  const defaultName = model.name;
  const version = spec?.checkpoint_model_name;
  const parameters = spec?.base_num_parameters;
  const contextSize = spec?.context_size;
  const isChatModel = spec?.is_chat;
  const description = descriptionOverride ?? model.description;

  return (
    <Stack gap="4">
      <Flex gap="2" align="center">
        <Text kind="body/semibold/xl" className="flex-1 truncate min-w-0">
          {defaultName}
        </Text>
        <Badge kind="solid" color="gray">
          <Flex align="center" gap="1">
            <Layers />
            {getModelVersion(version)}
          </Flex>
        </Badge>
      </Flex>
      <Flex align="center" gap="4" className="inline-flex [&_>.nv-divider-root]:flex-0">
        {status != null && (
          <>
            <Flex align="center" gap="1">
              <Tooltip slotContent={status}>
                <Flex align="center" gap="1">
                  <StatusIndicator
                    color={status === ModelDeploymentStatus.READY ? 'green' : 'red'}
                    size="small"
                  />
                  <Text kind="body/regular/sm">{status}</Text>
                </Flex>
              </Tooltip>
            </Flex>
            <Divider orientation="vertical" />
          </>
        )}
        {parameters != null && (
          <Tooltip slotContent="Parameters">
            <Flex align="center" gap="1">
              <HAlignLeft />
              <Text kind="body/regular/sm">
                {parametersToString(parameters, { format: 'short' })}
              </Text>
            </Flex>
          </Tooltip>
        )}
        {contextSize != null && (
          <Tooltip slotContent="Context Size">
            <Flex align="center" gap="1">
              <IconWindow />
              <Text kind="body/regular/sm">
                {parametersToString(contextSize, { format: 'short' })}
              </Text>
            </Flex>
          </Tooltip>
        )}
        {isChatModel && (
          <Tooltip slotContent="Chat Supported">
            <Flex align="center" gap="1">
              <ChatMulti />
              <Text kind="body/regular/sm">Chat</Text>
            </Flex>
          </Tooltip>
        )}
      </Flex>
      {badges && badges.length > 0 && (
        <Flex gap="2" align="center">
          {badges.map((tag) => (
            <Badge key={tag} color="gray" kind="solid">
              <Flex gap="1" align="center">
                {tagToIcon(tag)}
                <Text>{kebabCaseToTitleCase(tag)}</Text>
              </Flex>
            </Badge>
          ))}
        </Flex>
      )}
      {description && <Text kind="body/regular/md">{description}</Text>}
      {slotActions}
    </Stack>
  );
};

/**
 * Link to the customization job that produced a model or an adapter.
 *
 * Rendered beside the thing it describes, in the entity's own workspace: an adapter's run
 * is not its base model's run, and an adapter may live in a different workspace from its
 * base, so neither the section nor the workspace can be assumed from the model.
 */
const JobDetailsLink: FC<{ workspace: string; customizationJobId: string }> = ({
  workspace,
  customizationJobId,
}) => (
  <KVPair
    label="Job Details"
    value={
      <Anchor
        href={getWorkspaceCustomizationJobDetailsRoute(workspace, customizationJobId)}
        className="text-success"
      >
        View Job Details
      </Anchor>
    }
  />
);

/** Optional artifact/deployment fields not on ModelEntity (e.g. from deployment config or job API) */
export interface ModelArtifactData {
  backend_engine?: string;
  gpu_architecture?: string;
  tensor_parallelism?: number;
}

export type ModelParametersAccordionProps = {
  model: ModelEntity;
  /** When provided, shows a dedicated adapter details section */
  adapter?: Adapter | null;
  /** When available, deployment info for this model (provides status) */
  deployment?: ModelDeployment | null;
  /** Optional artifact fields from another source (e.g. deployment config); not on ModelEntity */
  artifactData?: ModelArtifactData | null;
  /** Hides Customizer-specific fields when the Customizer feature is disabled. */
  showCustomizationDetails?: boolean;
  /**
   * When available, ID of the customization job that produced `model` (enables "View Job
   * Details" in the Customization section). Always the model's own run.
   *
   * Callers scoped to an adapter leave this unset: the panel is then describing the
   * adapter, and the base model's run is not what the user asked about.
   */
  customizationJobId?: string | null;
  /**
   * When available, ID of the customization job that produced `adapter` (enables "View Job
   * Details" in the Adapter section). Resolve it from the adapter, never from `model`.
   */
  adapterJobId?: string | null;
};
export const ModelParametersAccordion = ({
  model,
  adapter,
  deployment,
  artifactData,
  showCustomizationDetails = true,
  customizationJobId,
  adapterJobId,
}: ModelParametersAccordionProps) => {
  const [isApiKeyVisible, setIsApiKeyVisible] = useState(false);
  const { spec } = model;
  const {
    context_size,
    checkpoint_model_name,
    family,
    base_num_parameters,
    minimum_gpus_all_weights,
    precision,
  } = spec ?? {};
  const { url, model_id, api_key } = model.api_endpoint ?? {};

  const inferenceAccordion = isDefined(model.api_endpoint)
    ? {
        chevronPosition: 'start' as const,
        slotTrigger: 'Inference',
        slotContent: (
          <Stack gap="4">
            {[
              { value: url, label: 'API Endpoint' },
              { value: model_id, label: 'Model ID' },
              ...(api_key
                ? [
                    {
                      value: isApiKeyVisible ? api_key : '••••••••••••••••••••••••••••••••••••••••',
                      label: 'API Key',
                      slotActions: () => (
                        <Flex align="center">
                          <CopyButton
                            text={api_key ?? ''}
                            color="neutral"
                            kind="tertiary"
                            size="tiny"
                          />
                          <Button
                            kind="tertiary"
                            size="tiny"
                            onClick={() => setIsApiKeyVisible(!isApiKeyVisible)}
                          >
                            {isApiKeyVisible ? <Eye /> : <EyeOff />}
                          </Button>
                        </Flex>
                      ),
                    },
                  ]
                : []),
            ].map((item) => (
              <CodeSnippetRoot
                className="[&_.nv-code-snippet-actions]:justify-between"
                key={item.label}
              >
                <CodeSnippetActions>
                  <Text kind="label/bold/sm">{item.label}</Text>
                  {item.slotActions ? (
                    item.slotActions()
                  ) : (
                    <CopyButton
                      text={item.value ?? ''}
                      color="neutral"
                      kind="tertiary"
                      size="tiny"
                    />
                  )}
                </CodeSnippetActions>
                <CodeSnippetCode value={item.value ?? ''} language="markdown" />
              </CodeSnippetRoot>
            ))}
          </Stack>
        ),
        value: 'inference',
      }
    : undefined;

  const isCustomizedModel =
    isDefined(model.base_model) ||
    isDefined(model.finetuning_type) ||
    (Array.isArray(model.adapters) && model.adapters.length > 0);

  const customizationAccordion =
    showCustomizationDetails && isCustomizedModel
      ? {
          chevronPosition: 'start' as const,
          slotTrigger: 'Customization Parameters',
          slotContent: (
            <Stack gap="2">
              {isDefined(model.created_at) && (
                <KVPair label="Created" value={<RelativeTime datetime={model.created_at} />} />
              )}
              <KVPair label="Base Model" value={model.base_model} />
              <KVPair
                label="Fine-tuning Type"
                value={
                  isDefined(model.finetuning_type)
                    ? formatFinetuningType(model.finetuning_type)
                    : undefined
                }
              />
              {isDefined(customizationJobId) && (
                <JobDetailsLink
                  workspace={model.workspace}
                  customizationJobId={customizationJobId}
                />
              )}
            </Stack>
          ),
          value: 'customization',
        }
      : undefined;

  const promptAccordion = isDefined(model.prompt)
    ? {
        chevronPosition: 'start' as const,
        slotTrigger: 'Prompt',
        slotContent: (
          <Stack gap="2">
            <KVPair
              label="Prompt"
              value={
                <ExpandableMessage
                  message={model.prompt.system_prompt}
                  characterLimit={175}
                  attributes={{ Text: { kind: 'body/semibold/md' } }}
                />
              }
            />
            <KVPair
              label="Few Shot Examples"
              value={
                <ExpandableMessage
                  message={model.prompt.icl_few_shot_examples}
                  characterLimit={175}
                  attributes={{ Text: { kind: 'body/semibold/md' } }}
                />
              }
            />
          </Stack>
        ),
        value: 'prompt',
      }
    : undefined;

  const showArtifact =
    model.fileset ||
    deployment?.status ||
    artifactData?.backend_engine ||
    artifactData?.gpu_architecture ||
    artifactData?.tensor_parallelism;
  const artifactAccordion = showArtifact
    ? {
        chevronPosition: 'start' as const,
        slotTrigger: 'Artifact Data',
        slotContent: (
          <Stack gap="2">
            <KVPair label="Files URL" value={model.fileset} />
            <KVPair label="Status" value={deployment?.status} />
            <KVPair label="Backend Engine" value={artifactData?.backend_engine} />
            <KVPair label="GPU Architecture" value={artifactData?.gpu_architecture} />
            <KVPair label="Precision" value={precision} />
            <KVPair
              label="Tensor Parallelism"
              value={
                isDefined(artifactData?.tensor_parallelism)
                  ? String(artifactData.tensor_parallelism)
                  : undefined
              }
            />
          </Stack>
        ),
        value: 'artifact',
      }
    : undefined;

  const modelParametersAccordion = {
    chevronPosition: 'start' as const,
    slotTrigger: 'Base Model Parameters',
    slotContent: (
      <Stack gap="2">
        <KVPair label="Creator" value={family ? kebabCaseToTitleCase(family) : undefined} />
        <KVPair label="Architecture" value="Transformer" />
        <KVPair label="Max I/O Tokens" value={context_size ? String(context_size) : undefined} />
        <KVPair
          label="Parameters"
          value={
            isDefined(base_num_parameters) ? parametersToString(base_num_parameters) : undefined
          }
        />
        {showCustomizationDetails && (
          <>
            <KVPair
              label="Fine-tune Options"
              value={
                isDefined(model.finetuning_type)
                  ? formatFinetuningType(model.finetuning_type)
                  : undefined
              }
            />
            <KVPair
              label="Recommended GPUs for Customization"
              value={
                isDefined(minimum_gpus_all_weights) ? String(minimum_gpus_all_weights) : undefined
              }
            />
          </>
        )}
        <KVPair label="Default Name" value={model.name} />
        <KVPair label="Version" value={checkpoint_model_name} />
      </Stack>
    ),
    value: 'model-parameters',
  };

  const adapterAccordion =
    showCustomizationDetails && adapter
      ? {
          chevronPosition: 'start' as const,
          slotTrigger: 'Adapter Details',
          slotContent: (
            <Stack gap="2">
              <KVPair label="Name" value={adapter.name} />
              {isDefined(adapter.description) && (
                <KVPair label="Description" value={adapter.description} />
              )}
              <KVPair
                label="Fine-tuning Type"
                value={formatFinetuningType(adapter.finetuning_type)}
              />
              <KVPair label="Fileset" value={adapter.fileset} />
              {isDefined(adapter.enabled) && (
                <KVPair label="Enabled" value={adapter.enabled ? 'Yes' : 'No'} />
              )}
              {isDefined(adapter.lora_config) && (
                <>
                  <KVPair label="LoRA Rank" value={String(adapter.lora_config.rank)} />
                  {isDefined(adapter.lora_config.alpha) && (
                    <KVPair label="LoRA Alpha" value={String(adapter.lora_config.alpha)} />
                  )}
                </>
              )}
              {isDefined(adapter.created_at) && (
                <KVPair label="Created" value={<RelativeTime datetime={adapter.created_at} />} />
              )}
              {isDefined(adapterJobId) && (
                <JobDetailsLink workspace={adapter.workspace} customizationJobId={adapterJobId} />
              )}
            </Stack>
          ),
          value: 'adapter',
        }
      : undefined;

  return (
    <Accordion
      multiple
      className="w-full border-t border-base"
      defaultValue={[
        'adapter',
        'inference',
        'prompt',
        'artifact',
        'customization',
        'model-parameters',
      ]}
      items={[
        adapterAccordion,
        inferenceAccordion,
        customizationAccordion,
        promptAccordion,
        artifactAccordion,
        modelParametersAccordion,
      ].filter(isDefined)}
    />
  );
};
