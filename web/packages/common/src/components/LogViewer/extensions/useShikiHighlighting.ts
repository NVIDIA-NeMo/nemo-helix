// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type Extension, RangeSetBuilder } from '@codemirror/state';
import {
  Decoration,
  type DecorationSet,
  EditorView,
  ViewPlugin,
  type ViewUpdate,
} from '@codemirror/view';
import { logger } from '@nemo/common/src/utils/logger';
import {
  type CodeSnippetLanguage,
  type CodeSnippetTheme,
  getHighlighter,
} from '@nvidia/foundations-react-core';
import { useEffect, useMemo, useState } from 'react';

type Highlighter = Awaited<ReturnType<typeof getHighlighter>>;
type ThemedToken = ReturnType<Highlighter['codeToTokensBase']>[number][number];

// Same themes and line limit as KUI's CodeSnippet.
const SHIKI_THEMES = {
  dark: 'github-dark-high-contrast',
  light: 'github-light-high-contrast',
} as const satisfies Record<'dark' | 'light', CodeSnippetTheme>;
const TOKENIZE_MAX_LINE_LENGTH = 1_000;

// Shiki's FontStyle bit flags.
const FONT_STYLE_ITALIC = 1;
const FONT_STYLE_BOLD = 2;

// Lines are cached by text so scrolling back over them doesn't re-tokenize. Log lines are
// mostly unique (timestamps), so cap the cache rather than let it grow with the log.
const MAX_CACHED_LINES = 2_000;

function shikiHighlighting(
  highlighter: Highlighter,
  language: CodeSnippetLanguage,
  themeName: CodeSnippetTheme
): Extension {
  const foreground = highlighter.getTheme(themeName).fg.toLowerCase();
  const marks = new Map<string, Decoration>();
  const tokenCache = new Map<string, ThemedToken[]>();

  const markFor = (color: string, fontStyle: number): Decoration => {
    const key = `${color}|${fontStyle}`;
    let mark = marks.get(key);
    if (!mark) {
      const styles = [`color: ${color}`];
      if (fontStyle > 0 && fontStyle & FONT_STYLE_ITALIC) styles.push('font-style: italic');
      if (fontStyle > 0 && fontStyle & FONT_STYLE_BOLD) styles.push('font-weight: bold');
      mark = Decoration.mark({ attributes: { style: styles.join('; ') } });
      marks.set(key, mark);
    }
    return mark;
  };

  // Lines are tokenized independently so only visible lines are tokenized, and a stray
  // quote in one line can't recolor every line after it.
  const tokenize = (text: string): ThemedToken[] => {
    let tokens = tokenCache.get(text);
    if (!tokens) {
      tokens =
        highlighter.codeToTokensBase(text, {
          lang: language,
          theme: themeName,
          tokenizeMaxLineLength: TOKENIZE_MAX_LINE_LENGTH,
        })[0] ?? [];
      if (tokenCache.size >= MAX_CACHED_LINES) tokenCache.clear();
      tokenCache.set(text, tokens);
    }
    return tokens;
  };

  const buildDecorations = (view: EditorView): DecorationSet => {
    const builder = new RangeSetBuilder<Decoration>();
    // Visible ranges can split one long line; never revisit a line, since the builder
    // only accepts ranges in order.
    let nextLineStart = 0;
    for (const { from, to } of view.visibleRanges) {
      for (let pos = Math.max(from, nextLineStart); pos <= to; pos = nextLineStart) {
        const line = view.state.doc.lineAt(pos);
        for (const token of tokenize(line.text)) {
          if (!token.content || !token.color || token.color.toLowerCase() === foreground) continue;
          const start = line.from + token.offset;
          builder.add(
            start,
            start + token.content.length,
            markFor(token.color, token.fontStyle ?? 0)
          );
        }
        nextLineStart = line.to + 1;
      }
    }
    return builder.finish();
  };

  const plugin = ViewPlugin.fromClass(
    class {
      decorations: DecorationSet;

      constructor(view: EditorView) {
        this.decorations = buildDecorations(view);
      }

      update(update: ViewUpdate) {
        if (update.docChanged || update.viewportChanged) {
          this.decorations = buildDecorations(update.view);
        }
      }
    },
    { decorations: (instance) => instance.decorations }
  );

  // Tokens in the theme's foreground aren't decorated, so the content carries that color.
  return [EditorView.theme({ '.cm-content': { color: foreground } }), plugin];
}

/** Highlights with KUI's shared Shiki highlighter; empty until grammar and themes load. */
export function useShikiHighlighting(
  language: CodeSnippetLanguage,
  theme: 'dark' | 'light'
): Extension {
  const [loaded, setLoaded] = useState<{
    highlighter: Highlighter;
    language: CodeSnippetLanguage;
  } | null>(null);

  useEffect(() => {
    let isMounted = true;
    const load = async () => {
      const highlighter = await getHighlighter();
      await Promise.all([
        highlighter.loadLanguage(language),
        highlighter.loadTheme(SHIKI_THEMES.dark, SHIKI_THEMES.light),
      ]);
      if (isMounted) setLoaded({ highlighter, language });
    };
    load().catch((error) => logger.error('Failed to load log syntax highlighting', error));
    return () => {
      isMounted = false;
    };
  }, [language]);

  return useMemo(
    () =>
      // Tokenizing with a grammar that hasn't loaded yet throws, so wait for this language.
      loaded?.language === language
        ? shikiHighlighting(loaded.highlighter, language, SHIKI_THEMES[theme])
        : [],
    [loaded, language, theme]
  );
}
