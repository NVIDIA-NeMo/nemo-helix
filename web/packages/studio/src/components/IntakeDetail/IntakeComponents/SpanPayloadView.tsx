// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CodeSnippet, Text } from '@nvidia/foundations-react-core';
import { PayloadPending } from '@studio/components/IntakeDetail/IntakeComponents/PayloadPending';
import { SpanPayloadChatView } from '@studio/components/IntakeDetail/IntakeComponents/SpanPayloadChatView';
import {
  autoFormat,
  parseChatPayload,
  parseJsonPayload,
  type SpanPayloadFormat,
} from '@studio/components/IntakeDetail/IntakeComponents/spanPayloadFormat';
import { type FC, lazy, Suspense, useEffect, useMemo, useState } from 'react';

const LARGE_PAYLOAD_RENDER_DEFER_CHAR_LIMIT = 20_000;

// ~100KB of react-markdown, for a format most readers never select.
const MarkdownContent = lazy(() =>
  import('@nemo/common/src/components/MarkdownContent').then((module) => ({
    default: module.MarkdownContent,
  }))
);

interface SpanPayloadViewProps {
  value: string | null | undefined;
  emptyMessage: string;
  format?: SpanPayloadFormat;
}

export const SpanPayloadView: FC<SpanPayloadViewProps> = ({ value, emptyMessage, format }) => {
  const payload = value && value.trim() ? value : null;
  const json = useMemo(() => parseJsonPayload(value), [value]);
  const chat = useMemo(() => parseChatPayload(value), [value]);
  // A caller can ask for a format on a payload that stopped supporting it.
  const unsupported = (format === 'json' && json === null) || (format === 'chat' && chat === null);
  const resolved = format && !unsupported ? format : autoFormat(!!json, !!chat);
  const text = resolved === 'json' && json !== null ? json : payload;

  // Very large payloads hold the main thread long enough to look blank.
  const shouldDeferRender = text !== null && text.length >= LARGE_PAYLOAD_RENDER_DEFER_CHAR_LIMIT;

  // Reset during render, not in an effect: an effect runs only after a commit
  // has already mounted the renderer with the new payload — the work deferral
  // exists to postpone. `resolved` counts, since identical text remounts anyway
  // when it moves between renderers.
  const [deferred, setDeferred] = useState(() => ({ text, resolved, show: !shouldDeferRender }));

  if (deferred.text !== text || deferred.resolved !== resolved) {
    setDeferred({ text, resolved, show: !shouldDeferRender });
  }

  useEffect(() => {
    if (deferred.show) {
      return;
    }
    // One committed paint with the spinner first. Render backpressure, not a
    // network loading state.
    const timeout = setTimeout(() => setDeferred((current) => ({ ...current, show: true })), 0);
    return () => clearTimeout(timeout);
  }, [deferred]);

  const showPayload = deferred.show && deferred.text === text && deferred.resolved === resolved;

  if (text === null) {
    return (
      <div className="flex min-h-[120px] items-center rounded-md border border-dashed border-base bg-surface-raised p-density-xl">
        <Text kind="body/regular/sm" className="text-secondary">
          {emptyMessage}
        </Text>
      </div>
    );
  }

  if (!showPayload) {
    return <PayloadPending />;
  }

  if (resolved === 'chat' && chat !== null) {
    return <SpanPayloadChatView messages={chat} />;
  }

  if (resolved === 'md') {
    return (
      <Suspense fallback={<PayloadPending />}>
        <div className="max-h-[420px] overflow-auto rounded-md border border-base bg-surface-raised p-density-xl">
          <MarkdownContent content={text} />
        </div>
      </Suspense>
    );
  }

  // Skip async Shiki highlighting on large payloads so the full text appears.
  const language = resolved === 'json' && !shouldDeferRender ? 'json' : 'text';

  return (
    <CodeSnippet
      // CodeSnippet keeps highlighted markup in state and never clears it, so
      // an unhighlighted language would still show the previous format's markup.
      key={language}
      value={text}
      language={language}
      kind="block"
      attributes={{
        CodeSnippetActions: { className: 'hidden' },
        CodeSnippetCode: {
          className:
            'max-h-[420px] [&_code]:whitespace-pre-wrap [&_code]:break-words [&_pre]:whitespace-pre-wrap',
        },
      }}
    />
  );
};
