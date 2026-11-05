// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { TraceData } from '@nemo/common/src/utils/TraceQueue';

export const formatTimestamp = (timestamp: number) => {
  return new Date(timestamp).toLocaleString();
};

export const makeReportTraceEmail = (trace: TraceData) => {
  const subject = `Trace Report: ${trace.severity.toUpperCase()} - ${trace.message}`;
  // note: using \ to add readability to the code without adding a new line to the body
  const body = `\
Hi,

I encountered an issue that needs investigation:

Trace ID: ${trace.traceId}
Timestamp: ${formatTimestamp(trace.timestamp)}
Severity: ${trace.severity.toUpperCase()}
Message: 
${trace.message}
${trace.context?.url ? `URL: ${trace.context.url}\n` : ''}
${trace.error ? `Error: ${trace.error.message}\n` : ''}

Spans: ${trace.spans.length} span${trace.spans.length === 1 ? '' : 's'} captured

Please use the Trace ID to look up the full trace details in the system.

Thanks!`;

  return `mailto:?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
};

export const makeReportAllTracesEmail = (traces: TraceData[], showErrorsOnly: boolean = false) => {
  // Filter traces based on showErrorsOnly flag
  const filteredTraces = showErrorsOnly ? traces.filter((t) => t.severity === 'error') : traces;

  const subject = showErrorsOnly
    ? `Error Trace Report: ${filteredTraces.length} error${filteredTraces.length === 1 ? '' : 's'} found`
    : `Trace Report: ${filteredTraces.length} trace${filteredTraces.length === 1 ? '' : 's'}`;

  const traceList = filteredTraces
    .map(
      (trace) =>
        `• ${trace.traceId} - [${trace.severity.toUpperCase()}] ${trace.message} (${formatTimestamp(trace.timestamp)})`
    )
    .join('\n');

  const body = `\
Hi,

I encountered ${showErrorsOnly ? 'error traces' : 'traces'} that need investigation:

${traceList}

Please use the Trace IDs above to look up the full trace details in the system.

Thanks!`;

  return `mailto:?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
};

export const formatTags = (tags: TraceData['spans'][number]['tags']) => {
  if (tags == null) {
    return 'empty';
  }

  if (typeof tags === 'object') {
    const firstKey = Object.keys(tags)[0];
    return `{ "${firstKey}": "${tags[firstKey]}", ... }`;
  }

  return JSON.stringify(tags);
};
