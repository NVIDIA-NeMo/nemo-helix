// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  TraceQueue,
  TraceData,
  TraceQueueEvent,
  TraceSpan,
} from '@nemo/common/src/utils/TraceQueue';
import { Modal, Button, Flex, Stack, Text, Badge, Switch } from '@nvidia/foundations-react-core';
import { CopyGeneric, Paperplane, ArrowLeft } from '@nvidia/react-gui-icons/icons';
import {
  formatTags,
  formatTimestamp,
  makeReportAllTracesEmail,
  makeReportTraceEmail,
} from '@studio/components/ReportTraceModal/utils';
import { useState, useEffect, useMemo } from 'react';

// for direct consumption by User, do not depend on the return type of this function
const serializableError = (error: object | undefined) => {
  const ret: Record<string, unknown> = {};

  if (error === null || error === undefined) {
    return ret;
  }
  if (typeof error !== 'object') {
    return ret;
  }

  const keys = Object.getOwnPropertyNames(error);

  for (const key of keys) {
    try {
      const value = error[key as keyof typeof error];
      ret[key] = value;
    } catch {
      // Skip properties that can't be accessed
      continue;
    }
  }

  return ret;
};

interface ReportTraceModalProps {
  open: boolean;
  onClose: () => void;
}

export const ReportTraceModal = ({ open, onClose }: ReportTraceModalProps) => {
  const [allTraces, setAllTraces] = useState<TraceData[]>([]);
  const [selectedTrace, setSelectedTrace] = useState<TraceData | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [showErrorsOnly, setShowErrorsOnly] = useState(true); // Default to showing errors only
  const toast = useToast();

  // Filter traces based on the toggle
  const traces = useMemo(() => {
    if (showErrorsOnly) {
      return allTraces.filter((trace) => trace.severity === 'error');
    }
    return allTraces;
  }, [allTraces, showErrorsOnly]);

  useEffect(() => {
    if (!open) return;

    const traceQueue = TraceQueue.getInstance();
    const currentTraces = traceQueue.getTraces();
    setAllTraces([...currentTraces]);

    // Subscribe to trace updates while modal is open
    const unsubscribe = traceQueue.subscribe((event: TraceQueueEvent) => {
      if (event.type === 'changed') {
        setAllTraces(event.traces);
      }
    });

    return () => {
      unsubscribe();
    };
  }, [open]);

  const handleCopyTrace = async (trace: TraceData) => {
    try {
      const traceData = {
        id: trace.id,
        traceId: trace.traceId,
        timestamp: new Date(trace.timestamp).toISOString(),
        severity: trace.severity,
        message: trace.message,
        context: trace.context,
        spans: trace.spans,
        error: serializableError(trace.error),
        metadata: trace.metadata,
      };

      await navigator.clipboard.writeText(JSON.stringify(traceData, null, 2));
      toast.success('Trace data copied to clipboard!');
    } catch {
      toast.error('Failed to copy trace data');
    }
  };

  const handleCopyAllTraces = async () => {
    try {
      const allTraces = traces.map((trace) => ({
        id: trace.id,
        traceId: trace.traceId,
        timestamp: new Date(trace.timestamp).toISOString(),
        severity: trace.severity,
        message: trace.message,
        context: trace.context,
        spans: trace.spans,
        error: serializableError(trace.error),
        metadata: trace.metadata,
      }));

      await navigator.clipboard.writeText(JSON.stringify(allTraces, null, 2));
      toast.success(`${traces.length} traces copied to clipboard!`);
    } catch {
      toast.error('Failed to copy trace data');
    }
  };

  const getSeverityColor = (severity: TraceData['severity']) => {
    switch (severity) {
      case 'error':
        return 'red';
      case 'warning':
        return 'yellow';
      case 'info':
        return 'blue';
      case 'debug':
        return 'gray';
      default:
        return 'gray';
    }
  };

  const handleViewDetails = (trace: TraceData) => {
    setSelectedTrace(trace);
    setShowDetails(true);
  };

  const handleBackToList = () => {
    setShowDetails(false);
    setSelectedTrace(null);
  };

  const handleClose = () => {
    setShowDetails(false);
    setSelectedTrace(null);
    setShowErrorsOnly(true);
    onClose();
  };

  const getFilterStatusText = () => {
    if (showErrorsOnly) {
      const errorCount = allTraces.filter((trace) => trace.severity === 'error').length;
      const totalCount = allTraces.length;
      return `Showing ${errorCount} error${errorCount === 1 ? '' : 's'} of ${totalCount} total trace${totalCount === 1 ? '' : 's'}`;
    } else {
      return `Showing all ${allTraces.length} trace${allTraces.length === 1 ? '' : 's'}`;
    }
  };

  return (
    <Modal
      open={open}
      onOpenChange={handleClose}
      slotHeading="Report a Trace"
      className="w-[90vw] max-w-[1000px] h-[80vh] max-h-[800px] flex flex-col"
      attributes={{
        ModalMain: { className: 'flex-grow-1' },
        ModalFooter: { className: 'justify-end' },
      }}
      slotFooter={
        <Flex gap="density-sm">
          <Button kind="tertiary" onClick={handleClose}>
            Close
          </Button>
          {!showDetails && (
            <>
              <Button
                color="brand"
                onClick={() => window.open(makeReportAllTracesEmail(traces, showErrorsOnly))}
                disabled={traces.length === 0}
              >
                <Paperplane />
                Email {showErrorsOnly ? 'Error' : 'All'} Traces
              </Button>
              <Button color="brand" onClick={handleCopyAllTraces} disabled={traces.length === 0}>
                <CopyGeneric />
                CopyGeneric {showErrorsOnly ? 'Error' : 'All'} Traces
              </Button>
            </>
          )}
        </Flex>
      }
    >
      <Stack gap="density-md" className="h-full overflow-hidden">
        {!showDetails ? (
          // Trace List View
          <>
            <Flex justify="between" align="center">
              <Text kind="body/regular/md">
                {allTraces.length === 0
                  ? 'No traces captured yet. Navigate around the application to generate telemetry traces.'
                  : getFilterStatusText()}
              </Text>

              {allTraces.length > 0 && (
                <Flex gap="density-sm" align="center">
                  <Text kind="body/regular/sm">Errors only</Text>
                  <Switch checked={showErrorsOnly} onCheckedChange={setShowErrorsOnly} />
                </Flex>
              )}
            </Flex>

            {traces.length > 0 && (
              <Stack gap="density-sm" className="overflow-auto flex-1">
                {traces.map((trace) => (
                  <Flex
                    key={trace.id}
                    justify="between"
                    align="center"
                    className="p-3 bg-surface-raised rounded-md border border-base"
                  >
                    <Button
                      kind="tertiary"
                      className="h-auto flex-1 justify-start p-0 text-left"
                      aria-label={`View details for ${trace.message}`}
                      onClick={() => handleViewDetails(trace)}
                    >
                      <Stack gap="density-xs">
                        <Flex gap="density-sm" align="center">
                          <Badge color={getSeverityColor(trace.severity)}>
                            {trace.severity.toUpperCase()}
                          </Badge>
                          <Text kind="label/semibold/md">{trace.message}</Text>
                        </Flex>
                        <Text kind="body/regular/sm" className="text-secondary">
                          {formatTimestamp(trace.timestamp)} • {trace.spans.length} span
                          {trace.spans.length === 1 ? '' : 's'}
                        </Text>
                        {trace.context?.url && (
                          <Text kind="body/regular/sm" className="text-secondary">
                            {trace.context.url}
                          </Text>
                        )}
                      </Stack>
                    </Button>
                    <Button kind="tertiary" size="small" onClick={() => handleCopyTrace(trace)}>
                      <CopyGeneric />
                    </Button>
                  </Flex>
                ))}
              </Stack>
            )}

            {showErrorsOnly && traces.length === 0 && allTraces.length > 0 && (
              <div className="text-center text-secondary p-5 bg-surface-raised rounded-md border border-base">
                <Text kind="body/regular/md">
                  No error traces found. Toggle off "Errors only" to see all {allTraces.length}
                  trace{allTraces.length === 1 ? '' : 's'}.
                </Text>
              </div>
            )}
          </>
        ) : (
          // Trace Details View (unchanged)
          selectedTrace && (
            <Stack gap="density-md" className="h-full overflow-hidden">
              <Flex justify="between" align="center">
                <Button kind="tertiary" onClick={handleBackToList}>
                  <ArrowLeft /> Back to List
                </Button>
                <Flex gap="density-sm">
                  <Button onClick={() => window.open(makeReportTraceEmail(selectedTrace))}>
                    <Paperplane />
                    Email Trace
                  </Button>
                  <Button onClick={() => handleCopyTrace(selectedTrace)}>
                    <CopyGeneric />
                    CopyGeneric Trace Data
                  </Button>
                </Flex>
              </Flex>

              <Stack gap="density-md" className="overflow-auto flex-1">
                <Stack gap="density-sm">
                  <Text kind="label/semibold/lg">Trace Information</Text>
                  <div className="p-3 bg-surface-raised rounded-md">
                    <Stack gap="density-xs">
                      <Text kind="body/regular/sm">
                        <strong>Trace ID:</strong> {selectedTrace.traceId}
                      </Text>
                      <Text kind="body/regular/sm">
                        <strong>Timestamp:</strong> {formatTimestamp(selectedTrace.timestamp)}
                      </Text>
                      <Text kind="body/regular/sm">
                        <strong>Severity:</strong> {selectedTrace.severity}
                      </Text>
                      <Text kind="body/regular/sm">
                        <strong>Message:</strong> {selectedTrace.message}
                      </Text>
                      {selectedTrace.context?.url && (
                        <Text kind="body/regular/sm">
                          <strong>URL:</strong> {selectedTrace.context.url}
                        </Text>
                      )}
                    </Stack>
                  </div>
                </Stack>

                <Stack gap="density-sm">
                  <Text kind="label/semibold/lg">Spans ({selectedTrace.spans.length})</Text>
                  <Stack gap="density-xs">
                    {selectedTrace.spans.map((span: TraceSpan) => (
                      <div key={span.spanId} className="p-2 bg-surface-raised rounded-md">
                        <Stack gap="density-xs">
                          <Text kind="body/semibold/sm">{span.operationName}</Text>
                          <Text kind="body/regular/sm" className="text-secondary">
                            Duration: {span.duration}ms | Span ID: {span.spanId}
                          </Text>
                          {span.parentSpanId && (
                            <Text kind="body/regular/sm" className="text-secondary">
                              Parent: {span.parentSpanId}
                            </Text>
                          )}
                          {span.tags && Object.keys(span.tags).length > 0 && (
                            <Text kind="body/regular/sm" className="text-secondary">
                              Tags: {formatTags(span.tags)}
                            </Text>
                          )}
                        </Stack>
                      </div>
                    ))}
                  </Stack>
                </Stack>

                {selectedTrace.error && (
                  <Stack gap="density-sm">
                    <Text kind="label/semibold/lg">Error Information</Text>
                    <div className="p-3 bg-feedback-danger rounded-md border border-feedback-danger">
                      <Stack gap="density-xs">
                        <Text kind="body/regular/sm">
                          <strong>Error:</strong> {selectedTrace.error.name}
                        </Text>
                        <Text kind="body/regular/sm">
                          <strong>Message:</strong> {selectedTrace.error.message}
                        </Text>
                        {selectedTrace.error.stack && (
                          <Text kind="body/regular/sm" className="font-mono text-xs">
                            <strong>Stack:</strong>
                            <br />
                            {selectedTrace.error.stack}
                          </Text>
                        )}
                      </Stack>
                    </div>
                  </Stack>
                )}
              </Stack>
            </Stack>
          )
        )}
      </Stack>
    </Modal>
  );
};
