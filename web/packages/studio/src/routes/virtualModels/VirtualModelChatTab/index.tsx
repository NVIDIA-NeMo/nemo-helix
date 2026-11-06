// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ParamsDropdown } from '@nemo/common/src/components/ModelSelectV2/ParamsDropdown';
import type { InferenceParams } from '@nemo/sdk/generated/platform/schema';
import { Block, Flex, Stack } from '@nvidia/foundations-react-core';
import {
  CHAT_INFERENCE_PARAM_FIELDS,
  chatParamMetadataWithMaxTokens,
  DEFAULT_CHAT_INFERENCE_PARAMS,
  mergeChatParams,
} from '@studio/components/chat/params';
import { ModelChat } from '@studio/components/ModelChat';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { type FC, useState } from 'react';

const PARAM_METADATA = chatParamMetadataWithMaxTokens(4096);

export const VirtualModelChatTab: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { virtualModelName } = useRequiredPathParams([ROUTE_PARAMS.virtualModelName]);

  const [inferenceParams, setInferenceParams] = useState<InferenceParams>({
    ...DEFAULT_CHAT_INFERENCE_PARAMS,
    max_tokens: PARAM_METADATA.max_tokens.default,
  });

  return (
    <Stack className="min-h-0 flex-1">
      <Flex align="center" justify="end" className="shrink-0 border-b border-base pb-3">
        <ParamsDropdown
          inferenceParams={inferenceParams}
          onInferenceParamsChange={(next) =>
            setInferenceParams((prev) => mergeChatParams(prev, next))
          }
          fields={CHAT_INFERENCE_PARAM_FIELDS}
          fieldMetadata={PARAM_METADATA}
        />
      </Flex>
      <Block className="h-full min-h-0 pt-4">
        <ModelChat
          key={`${workspace}/${virtualModelName}`}
          model={virtualModelName}
          workspace={workspace}
          assistantName={virtualModelName}
          promptData={{ inference_params: inferenceParams }}
        />
      </Block>
    </Stack>
  );
};
