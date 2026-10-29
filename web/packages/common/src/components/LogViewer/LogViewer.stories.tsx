// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LogViewer } from '@nemo/common/src/components/LogViewer';
import { ToastProvider } from '@nemo/common/src/providers/toast/ToastProvider';
import type { LogLoadProgress } from '@nemo/common/src/utils/logs';
import type { HelixJobLog } from '@nemo/sdk/generated/platform/schema';
import {
  Flex,
  ThemeProvider as KaizenThemeProvider,
  Panel,
  type Theme,
} from '@nvidia/foundations-react-core';
import type { Decorator, Meta, StoryObj } from '@storybook/react';
import { type ComponentProps, type FC, useEffect, useMemo, useState } from 'react';

const LOG_START_MS = Date.parse('2026-01-01T00:00:00Z');
const LOG_INTERVAL_MS = 250;

const LEVELS = ['INFO', 'INFO', 'INFO', 'DEBUG', 'WARNING', 'ERROR'] as const;
const MESSAGES = [
  'Loading checkpoint shards from /models/llama-3.1-8b-instruct',
  'loss=1.2841 lr=2.0e-05 grad_norm=0.842 tokens_per_sec=18234',
  'Saving checkpoint to "/outputs/checkpoint-latest"',
  'Evaluating on validation split (512 samples)',
  'CUDA memory allocated: 38.2 GiB / 80.0 GiB',
  "Retrying request to inference gateway (attempt 2/5): 'connection reset by peer'",
  'Traceback (most recent call last): File "/app/train.py", line 214, in <module>',
];
const LONG_LINE_PAYLOAD = JSON.stringify({
  model: 'meta/llama-3.1-8b-instruct',
  dataset: 'default/customization-sample-dataset',
  hyperparameters: { epochs: 3, batch_size: 16, learning_rate: 2e-5, lora_rank: 16 },
  output: 's3://nemo-helix-artifacts/customizations/default/job-demo/checkpoints/latest',
});
const MULTILINE_TRACEBACK = [
  'Traceback (most recent call last):',
  '  File "/app/train.py", line 214, in <module>',
  '    main()',
  '  File "/app/train.py", line 198, in main',
  '    trainer.train()',
  'RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB',
].join('\n');

interface LogOptions {
  /** Append a long JSON payload to every line to exercise horizontal scroll and wrapping. */
  longLines: boolean;
  /** Make every 25th message a multi-line traceback, as Python jobs emit. */
  multiline: boolean;
  /** End messages with `\r`, as some container runtimes do, so the text has `\r\n` breaks. */
  carriageReturns: boolean;
}

/** Deterministic so screenshots stay stable between reloads. */
const makeLog = (index: number, options: LogOptions): HelixJobLog => {
  const message =
    options.multiline && index % 25 === 24
      ? `ERROR   step=${index} ${MULTILINE_TRACEBACK}`
      : `${LEVELS[index % LEVELS.length].padEnd(7)} step=${index} ${
          MESSAGES[index % MESSAGES.length]
        }${options.longLines ? ` ${LONG_LINE_PAYLOAD}` : ''}`;
  return {
    timestamp: new Date(LOG_START_MS + index * LOG_INTERVAL_MS).toISOString(),
    job: 'job-demo',
    job_step: 'customization-training',
    job_task: 'task-0',
    message: options.carriageReturns ? `${message}\r` : message,
  };
};

const makeLogs = (count: number, options: LogOptions): HelixJobLog[] =>
  Array.from({ length: count }, (_, index) => makeLog(index, options));

/** Appends one more line every `intervalMs`; 0 disables streaming. */
const useStreamedCount = (intervalMs: number) => {
  const [count, setCount] = useState(0);
  useEffect(() => {
    if (intervalMs <= 0) return;
    const timer = setInterval(() => setCount((prev) => prev + 1), intervalMs);
    return () => clearInterval(timer);
  }, [intervalMs]);
  return count;
};

interface StoryArgs
  extends Omit<ComponentProps<typeof LogViewer>, 'logs' | 'onNotify'>, LogOptions {
  /** Number of generated log lines. */
  lineCount: number;
  /** Append a new line on this interval to exercise auto-scroll; 0 disables streaming. */
  streamIntervalMs: number;
}

const GeneratedLogViewer: FC<StoryArgs> = ({
  lineCount,
  streamIntervalMs,
  longLines,
  multiline,
  carriageReturns,
  ...props
}) => {
  const streamedCount = useStreamedCount(streamIntervalMs);
  const logs = useMemo(
    () => makeLogs(lineCount + streamedCount, { longLines, multiline, carriageReturns }),
    [lineCount, streamedCount, longLines, multiline, carriageReturns]
  );
  return <LogViewer {...props} logs={logs} />;
};

/** Walks a log a page at a time like the job detail routes, then shows it. */
const PageWalkLogViewer: FC<StoryArgs> = ({ lineCount, ...args }) => {
  const pageSize = Math.max(1, Math.ceil(lineCount / 10));
  const [progress, setProgress] = useState<LogLoadProgress | null>(null);

  useEffect(() => {
    let loaded = 0;
    const timer = setInterval(() => {
      loaded = Math.min(lineCount, loaded + pageSize);
      setProgress({ loaded, total: lineCount });
      if (loaded >= lineCount) clearInterval(timer);
    }, 400);
    return () => clearInterval(timer);
  }, [lineCount, pageSize]);

  const isLoading = !progress || progress.loaded < lineCount;
  return (
    <GeneratedLogViewer
      {...args}
      lineCount={isLoading ? 0 : lineCount}
      isLoading={isLoading}
      loadProgress={progress}
    />
  );
};

/** `fillHeight` stretches to its container, as in the customization job's Logs tab. */
const withFillContainer: Decorator = (Story) => (
  <Flex className="h-[560px] min-h-0 w-full">
    <Story />
  </Flex>
);

/** Most job detail routes render the viewer inside a compact Panel. */
const withPanel: Decorator = (Story) => (
  <Panel slotHeading="Logs" elevation="high" density="compact">
    <Story />
  </Panel>
);

const meta = {
  title: 'Studio Common/LogViewer',
  render: (args) => <GeneratedLogViewer {...args} />,
  decorators: [
    // CodeMirror picks its theme from KUI's useTheme, which needs a ThemeProvider.
    (Story, context) => {
      const sbTheme = (context.globals.theme === 'light' ? 'light' : 'dark') as Theme;
      return (
        <KaizenThemeProvider density="standard" global={false} theme={sbTheme}>
          <ToastProvider>
            <Story />
          </ToastProvider>
        </KaizenThemeProvider>
      );
    },
  ],
  args: {
    lineCount: 200,
    longLines: false,
    multiline: false,
    carriageReturns: false,
    streamIntervalMs: 0,
    isLoading: false,
    fillHeight: false,
    rows: 30,
    downloadFilename: 'job-demo-logs.txt',
    emptyMessage: 'No logs available yet',
    logStatus: 'ended',
  },
  argTypes: {
    lineCount: { control: { type: 'number', min: 0, step: 100 } },
    streamIntervalMs: { control: { type: 'number', min: 0, step: 100 } },
    rows: { control: { type: 'number', min: 1 } },
    loadProgress: { control: 'object' },
    logStatus: { control: 'select', options: ['live', 'ended', undefined] },
  },
} satisfies Meta<StoryArgs>;

export default meta;
type Story = StoryObj<typeof meta>;

/** Tail of a longer log, with "Load previous logs" to reveal the rest. */
export const Default: Story = {};

export const InPanel: Story = {
  decorators: [withPanel],
};

/** No "Load previous logs" when everything already fits. */
export const FewerLinesThanRows: Story = {
  args: { lineCount: 10 },
};

export const SingleLine: Story = {
  args: { lineCount: 1 },
};

/** Agent deployment logs show a taller tail. */
export const CustomRows: Story = {
  args: { rows: 40 },
};

/** Lines overflow horizontally until "Wrap lines" is toggled on. */
export const LongLines: Story = {
  args: { longLines: true },
};

/** Messages with embedded newlines span several editor lines but count as one log line. */
export const MultilineMessages: Story = {
  args: { multiline: true, lineCount: 100 },
};

export const CarriageReturns: Story = {
  args: { carriageReturns: true, streamIntervalMs: 500, logStatus: 'live' },
};

export const FillHeight: Story = {
  args: { fillHeight: true, lineCount: 500 },
  decorators: [withFillContainer],
};

export const FillHeightLongLines: Story = {
  args: { fillHeight: true, lineCount: 500, longLines: true },
  decorators: [withFillContainer],
};

/** New lines keep the view pinned to the bottom until the reader scrolls up. */
export const Streaming: Story = {
  args: { lineCount: 100, streamIntervalMs: 500, logStatus: 'live' },
};

export const StreamingFillHeight: Story = {
  args: { lineCount: 100, streamIntervalMs: 500, fillHeight: true, logStatus: 'live' },
  decorators: [withFillContainer],
};

/** Streams into an empty log, as a freshly started deployment does. */
export const StreamingFromEmpty: Story = {
  args: {
    lineCount: 0,
    streamIntervalMs: 750,
    rows: 40,
    logStatus: 'live',
    emptyMessage: 'No log output yet — the deployment may not have started writing.',
  },
};

export const LargeLog: Story = {
  args: { lineCount: 50_000, fillHeight: true },
  decorators: [withFillContainer],
};

export const NoDownload: Story = {
  args: { downloadFilename: undefined },
};

/** Callers that fetch in one shot omit `loadProgress`, leaving a bare spinner. */
export const Loading: Story = {
  args: { isLoading: true },
};

/** The total isn't known until the first page lands. */
export const LoadingBeforeTotal: Story = {
  args: { isLoading: true, loadProgress: null },
};

export const LoadingWithProgress: Story = {
  args: { isLoading: true, loadProgress: { loaded: 12_000, total: 48_000 } },
};

/** Progress counts climb page by page, then the logs replace the spinner. */
export const PageWalk: Story = {
  args: { lineCount: 5_000 },
  render: (args) => <PageWalkLogViewer {...args} />,
  decorators: [withPanel],
};

export const Empty: Story = {
  args: { lineCount: 0 },
};

export const EmptyCustomMessage: Story = {
  args: { lineCount: 0, emptyMessage: 'Waiting for the job to emit its first log lines...' },
  decorators: [withPanel],
};
