// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ActionBarPrimitive, MessagePrimitive } from '@assistant-ui/react';
import { useMessageError } from '@nemo/common/src/components/AssistantChat/useMessageError';
import { useCopyToClipboard } from '@nemo/common/src/hooks/useCopyToClipboard';
import { Check, CopyGeneric } from '@nemo/icons/icons';
import { Tooltip } from '@nvidia/foundations-react-core';
import { useEffect, useState } from 'react';

export const ACTION_BUTTON_CLASS =
  'flex cursor-pointer size-7 items-center justify-center rounded border border-transparent text-base bg-transparent text-secondary hover:border-base hover:bg-surface-raised hover:text-primary disabled:cursor-not-allowed disabled:opacity-50';

export const MESSAGE_ACTIONS_CLASS =
  'flex gap-density-xs opacity-0 transition-opacity group-hover/message:opacity-100 group-focus-within/message:opacity-100 [@media(hover:none)]:opacity-100';

const COPIED_DURATION_MS = 3000;

interface CopyErrorActionProps {
  error: string;
}

type CopyState = 'idle' | 'copied' | 'failed';

// A failed reply's body holds the human-readable explanation, so its copy button copies the raw
// error (status and provider detail) instead; that is what gets pasted into a bug report.
// Writing can fail: `navigator.clipboard` is missing on pages that are not a secure context
// (plain http outside localhost) and `writeText` rejects when the browser denies clipboard access
// or the document is not focused. A failure is shown in the tooltip instead of a false check mark.
const CopyErrorAction = ({ error }: CopyErrorActionProps) => {
  const [copyState, setCopyState] = useState<CopyState>('idle');
  const { copyToClipboard } = useCopyToClipboard({
    onSuccess: () => setCopyState('copied'),
    onError: () => setCopyState('failed'),
  });

  useEffect(() => {
    if (copyState === 'idle') return undefined;
    const timeout = setTimeout(() => setCopyState('idle'), COPIED_DURATION_MS);
    return () => clearTimeout(timeout);
  }, [copyState]);

  return (
    <Tooltip
      slotContent={
        copyState === 'failed' ? 'Could not copy: clipboard access was blocked' : 'Copy error'
      }
    >
      <button
        type="button"
        aria-label="Copy error"
        className={ACTION_BUTTON_CLASS}
        onClick={() => void copyToClipboard(error)}
      >
        {copyState === 'copied' ? <Check size="16" /> : <CopyGeneric size="16" />}
      </button>
    </Tooltip>
  );
};

export const CopyAction = () => {
  const error = useMessageError();
  if (error) return <CopyErrorAction error={error} />;

  return (
    <Tooltip slotContent="Copy message">
      <ActionBarPrimitive.Copy aria-label="Copy message" className={ACTION_BUTTON_CLASS}>
        <MessagePrimitive.If copied>
          <Check size="16" />
        </MessagePrimitive.If>
        <MessagePrimitive.If copied={false}>
          <CopyGeneric size="16" />
        </MessagePrimitive.If>
      </ActionBarPrimitive.Copy>
    </Tooltip>
  );
};
