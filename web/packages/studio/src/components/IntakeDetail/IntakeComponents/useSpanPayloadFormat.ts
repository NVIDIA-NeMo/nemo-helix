// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  autoFormat,
  readPayloadFormats,
  type SpanPayloadFormat,
  type SpanPayloadFormatState,
} from '@studio/components/IntakeDetail/IntakeComponents/spanPayloadFormat';
import { useCallback, useMemo, useState } from 'react';

/**
 * Call this in the component owning both slots of an accordion item.
 *
 * @param onSelect Runs when a format is picked — e.g. to open the section,
 *   since the toggle sits on a trigger that may be collapsed.
 */
export const useSpanPayloadFormat = (
  value: string | null | undefined,
  onSelect?: () => void
): SpanPayloadFormatState => {
  const { isJson, isChat } = useMemo(() => readPayloadFormats(value), [value]);

  // Keyed by the payload it was made for, so different text re-derives the
  // default instead of keeping a JSON view it cannot satisfy.
  const [selection, setSelection] = useState<{
    value: string | null | undefined;
    format: SpanPayloadFormat;
  } | null>(null);

  const select = useCallback(
    (format: SpanPayloadFormat) => {
      setSelection({ value, format });
      onSelect?.();
    },
    [value, onSelect]
  );

  return {
    format: selection && selection.value === value ? selection.format : autoFormat(isJson, isChat),
    select,
    isJson,
    isChat,
    isEmpty: !value?.trim(),
  };
};
