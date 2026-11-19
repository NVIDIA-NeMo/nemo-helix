// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TraceQueue } from '@nemo/common/src/utils/TraceQueue';
import type { ReadableSpan, SpanProcessor } from '@opentelemetry/sdk-trace-web';
import {
  enhanceFetchSpanName,
  enhanceXhrSpanName,
  enhanceClickSpanName,
  enhanceNavigationSpanName,
  enhanceSubmitSpanName,
  hrTimeToMilliseconds,
} from '@studio/telemetry/util';

/**
 * SpanProcessor that captures OpenTelemetry spans and adds them to our TraceQueue
 * for the "Report a Trace" functionality.
 *
 * ## Common Span Names from Auto-Instrumentations:
 *
 * ### Navigation Spans (@opentelemetry/instrumentation-user-interaction):
 * - `"Navigation: /route/path"` - Route navigation events (e.g., "Navigation: /projects/123")
 * - `"click"` - User click interactions
 * - `"submit"` - Form submission events
 *
 * ### HTTP Request Spans:
 * - `"HTTP GET"`, `"HTTP POST"`, `"HTTP PUT"`, `"HTTP DELETE"` - From @opentelemetry/instrumentation-fetch
 * - `"GET"`, `"POST"`, `"PUT"`, `"DELETE"` - From @opentelemetry/instrumentation-xml-http-request
 *
 * ### Other Potential Spans:
 * - Custom spans from manual instrumentation
 * - Resource loading spans (images, scripts, etc.)
 * - Performance measurement spans
 *
 * ## Enhanced Span Names:
 * This processor enhances span names to provide better context for users:
 * - `"HTTP GET"` → `"FETCH GET /api/path"`
 * - `"GET"` → `"XHR GET /api/path"`
 * - `"click"` → `"Click on button#submit"` or `"Click on div.card"`
 * - `"Navigation: /path"` → `"Navigation"`
 * - `"submit"` → `"Form Submit #formId"`
 *
 * The original span names are preserved in the span data for debugging purposes.
 */
export class TraceQueueSpanProcessor implements SpanProcessor {
  private traceQueue: TraceQueue;

  constructor() {
    this.traceQueue = TraceQueue.getInstance();
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }

  onStart(): void {
    // Nothing to do on start
  }

  /**
   * Enhances the span name to provide better context for different types of operations.
   */
  private enhanceSpanName(span: ReadableSpan): string {
    const originalName = span.name;
    const attributes = span.attributes;

    // Handle HTTP requests - distinguish between FETCH and XHR
    if (originalName.startsWith('HTTP ')) {
      return enhanceFetchSpanName(originalName, attributes);
    }

    if (['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'].includes(originalName)) {
      return enhanceXhrSpanName(originalName, attributes);
    }

    // Handle click events - add context about the clicked element
    if (originalName === 'click') {
      return enhanceClickSpanName(attributes);
    }

    // Handle navigation events - keep them clean
    if (originalName.startsWith('Navigation: ')) {
      return enhanceNavigationSpanName();
    }

    // Handle form submissions
    if (originalName === 'submit') {
      return enhanceSubmitSpanName(attributes);
    }

    // Return original name for other spans
    return originalName;
  }

  onEnd(span: ReadableSpan): void {
    // Convert OpenTelemetry span to TraceSpan and add to queue
    const traceSpan = {
      traceId: span.spanContext().traceId,
      spanId: span.spanContext().spanId,
      parentSpanId: span.parentSpanContext?.spanId,
      operationName: this.enhanceSpanName(span), // Use enhanced span name
      startTime: hrTimeToMilliseconds(span.startTime),
      endTime: hrTimeToMilliseconds(span.endTime),
      duration: hrTimeToMilliseconds(span.endTime) - hrTimeToMilliseconds(span.startTime),
      tags: span.attributes,
      status: {
        code: span.status.code,
        message: span.status.message,
      },
    };

    // Use the new addSpan method which will handle trace grouping automatically
    this.traceQueue.addSpan(traceSpan);
  }

  shutdown(): Promise<void> {
    return Promise.resolve();
  }
}
