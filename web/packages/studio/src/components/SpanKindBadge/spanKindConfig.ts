// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SpanKind } from '@nemo/sdk/generated/platform/schema';
import type { BadgeProps } from '@nvidia/foundations-react-core';
import {
  ArrowUpDown,
  Bot,
  Boxes,
  CircleHelp,
  ClipboardCheck,
  Link2,
  Search,
  ShieldCheck,
  Sparkles,
  Wrench,
  type LucideIcon,
} from 'lucide-react';

type BadgeColor = Exclude<BadgeProps['color'], null | undefined>;

export interface SpanKindConfig {
  label: string;
  color: BadgeColor;
  icon: LucideIcon;
}

/** Span-kind → badge color/icon/label, mirroring the Experiments design. */
const SPAN_KIND_CONFIG: Record<SpanKind, SpanKindConfig> = {
  [SpanKind.AGENT]: { label: 'Agent', color: 'teal', icon: Bot },
  [SpanKind.LLM]: { label: 'LLM', color: 'purple', icon: Sparkles },
  [SpanKind.TOOL]: { label: 'Tool', color: 'blue', icon: Wrench },
  [SpanKind.RETRIEVER]: { label: 'Retriever', color: 'blue', icon: Search },
  [SpanKind.EMBEDDING]: { label: 'Embedding', color: 'blue', icon: Boxes },
  [SpanKind.RERANKER]: { label: 'Reranker', color: 'blue', icon: ArrowUpDown },
  [SpanKind.EVALUATOR]: { label: 'Evaluator', color: 'green', icon: ClipboardCheck },
  [SpanKind.GUARDRAIL]: { label: 'Guardrail', color: 'yellow', icon: ShieldCheck },
  [SpanKind.CHAIN]: { label: 'Chain', color: 'gray', icon: Link2 },
  [SpanKind.UNKNOWN]: { label: 'Unknown', color: 'gray', icon: CircleHelp },
};

const FALLBACK_CONFIG: SpanKindConfig = SPAN_KIND_CONFIG[SpanKind.UNKNOWN];

/** Resolves the color/icon/label for a span kind, falling back to "Unknown". */
export const getSpanKindConfig = (kind: SpanKind | string | undefined): SpanKindConfig =>
  (kind && SPAN_KIND_CONFIG[kind as SpanKind]) || FALLBACK_CONFIG;

/**
 * Badge color → icon text color. Shared by every surface that renders a span
 * kind (badge, tree, graph card) so a kind reads the same color everywhere.
 */
const KIND_ICON_COLOR_CLASS: Record<string, string> = {
  teal: 'text-[color:var(--text-color-accent-teal)]',
  purple: 'text-[color:var(--text-color-accent-purple)]',
  blue: 'text-[color:var(--text-color-accent-blue)]',
  green: 'text-[color:var(--text-color-accent-green)]',
  yellow: 'text-[color:var(--text-color-accent-yellow)]',
  red: 'text-[color:var(--text-color-accent-red)]',
  gray: 'text-[color:var(--text-color-secondary)]',
};

/** Icon/accent text color class for a span kind. */
export const getSpanKindColorClass = (kind: SpanKind | string | undefined): string =>
  KIND_ICON_COLOR_CLASS[getSpanKindConfig(kind).color] ?? KIND_ICON_COLOR_CLASS.gray;
