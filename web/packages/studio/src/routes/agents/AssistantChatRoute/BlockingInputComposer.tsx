// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  AgentSelectBlockingInput,
  EvalConfigBlockingInput,
  FilesetFileBlockingInput,
  ModelSelectBlockingInput,
  type AgentBlockingInputStatus,
  type AgentBlockingInputSubmission,
} from '@studio/components/agents/AgentBlockingInput';
import { getBlockingInputRequest } from '@studio/routes/agents/AssistantChatRoute/blockingInputRequest';
import type { AssistantInputRequest } from '@studio/routes/agents/AssistantChatRoute/types';
import type { FC } from 'react';

interface BlockingInputComposerProps {
  readonly inputRequest: AssistantInputRequest;
  readonly inputStatus: AgentBlockingInputStatus;
  readonly workspace: string;
  readonly onSubmit: (submission: AgentBlockingInputSubmission) => Promise<void> | void;
  readonly onSkip: () => Promise<void> | void;
}

const DEFAULT_DATASET_FILE_TYPES = ['.json', '.jsonl', '.csv', '.parquet'] as const;

export const BlockingInputComposer: FC<BlockingInputComposerProps> = ({
  inputRequest,
  inputStatus,
  workspace,
  onSubmit,
  onSkip,
}) => {
  const request = getBlockingInputRequest(inputRequest);
  const requestKey = request.id;
  const sharedProps = {
    input: inputRequest.input,
    request,
    status: inputStatus,
    onSubmit,
    onSkip,
  };

  switch (inputRequest.kind) {
    case 'agent':
      return <AgentSelectBlockingInput key={requestKey} {...sharedProps} workspace={workspace} />;
    case 'eval_config':
      return <EvalConfigBlockingInput key={requestKey} {...sharedProps} workspace={workspace} />;
    case 'dataset_file':
      return (
        <FilesetFileBlockingInput
          key={requestKey}
          {...sharedProps}
          defaultAcceptedFileTypes={[...DEFAULT_DATASET_FILE_TYPES]}
          missingSelectionMessage="Pick a dataset file inside an existing fileset"
          selectionDisplayLabel="Selected dataset"
          submitLabel="Select dataset"
          toValue={({ name, objectPath }) => ({
            dataset_fileset: name,
            dataset_path: objectPath,
          })}
          workspace={workspace}
        />
      );
    case 'model':
      return <ModelSelectBlockingInput key={requestKey} {...sharedProps} />;
    default: {
      const exhaustiveCheck: never = inputRequest.kind;
      return exhaustiveCheck;
    }
  }
};
