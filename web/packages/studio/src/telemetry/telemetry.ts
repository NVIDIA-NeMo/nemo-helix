// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type LogRecord, logs } from '@opentelemetry/api-logs';
import { getWebAutoInstrumentations } from '@opentelemetry/auto-instrumentations-web';
import { ZoneContextManager } from '@opentelemetry/context-zone';
import { OTLPLogExporter } from '@opentelemetry/exporter-logs-otlp-http';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { registerInstrumentations } from '@opentelemetry/instrumentation';
import { resourceFromAttributes } from '@opentelemetry/resources';
import {
  LoggerProvider,
  BatchLogRecordProcessor,
  ConsoleLogRecordExporter,
  SimpleLogRecordProcessor,
} from '@opentelemetry/sdk-logs';
import {
  BatchSpanProcessor,
  type SpanProcessor,
  WebTracerProvider,
} from '@opentelemetry/sdk-trace-web';
import { ATTR_SERVICE_NAME } from '@opentelemetry/semantic-conventions';
import {
  OTEL_PROXY_URL,
  OTEL_SERVICE_NAME,
  TELEMETRY_ENABLED,
  isLocalDevelopmentEnv,
} from '@studio/constants/environment';
import { TraceQueueSpanProcessor } from '@studio/telemetry/TraceQueueSpanProcessor';

/**
 * Log Exporter used for local development that simply logs a record's message to the
 * console. By default, an OpenTelemetry log is an object with additional metadata
 * that may be useful in deployed environments. Locally though, we're just interested
 * in logging the actual message.
 */
class SimpleConsoleLogExporter extends ConsoleLogRecordExporter {
  export(logRecords: LogRecord[]): void {
    logRecords.forEach((record) => {
      // eslint-disable-next-line no-console
      console.log(record.body);
    });
  }
}

/**
 * Instantiates and registers an OpenTelemetry Logger Provider for Studio UI. For local development,
 * logs are just exported to the console. In deployed environments, logs are also exported to our
 * OpenTelemetry service endpoint.
 *
 * To log information, see `logger.ts` for a global instance of the logger that components can import.
 */
export const registerLoggerProvider = (): LoggerProvider => {
  const processors = isLocalDevelopmentEnv
    ? [new SimpleLogRecordProcessor(new SimpleConsoleLogExporter())]
    : TELEMETRY_ENABLED
      ? [new BatchLogRecordProcessor(new OTLPLogExporter({ url: `${OTEL_PROXY_URL}/v1/logs` }))]
      : [];

  const loggerProvider = new LoggerProvider({
    resource: resourceFromAttributes({
      [ATTR_SERVICE_NAME]: OTEL_SERVICE_NAME,
    }),
    processors,
  });

  logs.setGlobalLoggerProvider(loggerProvider);
  return loggerProvider;
};

/**
 * Instantiates and registers an OpenTelemetry Tracer Provider for Studio UI. The Provider handles distributed
 * tracing for the UI by collecting and exporting spans (units of work that represent a request or action performed
 * by the user) to our OpenTelemetry service endpoint.
 */
export const registerTelemetryProvider = (): {
  provider: WebTracerProvider;
  unload: () => void;
} => {
  const spanProcessors: SpanProcessor[] = [new TraceQueueSpanProcessor()];

  // For deployed environments, also export traces to our proxy endpoint
  if (!isLocalDevelopmentEnv && TELEMETRY_ENABLED) {
    const traceExporter = new OTLPTraceExporter({ url: `${OTEL_PROXY_URL}/v1/traces` });
    spanProcessors.push(new BatchSpanProcessor(traceExporter));
  }

  const provider = new WebTracerProvider({
    resource: resourceFromAttributes({
      [ATTR_SERVICE_NAME]: OTEL_SERVICE_NAME,
    }),
    spanProcessors,
  });

  // Registers our tracer provider. We use `ZoneContextManager` so OpenTelemetry can track asynchronous operations.
  provider.register({
    contextManager: new ZoneContextManager(),
  });

  // Registers auto-instrumentations with our tracer provider, which collects common telemetry data out-of-the-box.
  const unload = registerInstrumentations({
    tracerProvider: provider,
    instrumentations: getWebAutoInstrumentations(),
  });

  return { provider, unload };
};

let activeLoggerProvider: LoggerProvider | null = null;
let activeTracerProvider: WebTracerProvider | null = null;
let unloadInstrumentations: (() => void) | null = null;
let isRegistered = false;

function initializeTelemetry() {
  if (isRegistered) return;

  activeLoggerProvider = registerLoggerProvider();

  if (TELEMETRY_ENABLED) {
    const { provider, unload } = registerTelemetryProvider();
    activeTracerProvider = provider;
    unloadInstrumentations = unload;
  }

  isRegistered = true;
}

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    unloadInstrumentations?.();
    void activeTracerProvider?.shutdown();
    void activeLoggerProvider?.shutdown();
    unloadInstrumentations = null;
    activeTracerProvider = null;
    activeLoggerProvider = null;
    isRegistered = false;
  });

  import.meta.hot.accept(() => {
    initializeTelemetry();
  });
}

initializeTelemetry();
