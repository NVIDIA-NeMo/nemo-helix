// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type Extension, StateEffect } from '@codemirror/state';
import { type EditorView, ViewPlugin, type ViewUpdate } from '@codemirror/view';

const AT_BOTTOM_THRESHOLD_PX = 50;

/** Jump to the bottom and re-arm auto-scroll. */
export const pinToBottom = StateEffect.define<null>();

/**
 * Keeps the editor pinned to the bottom as lines arrive while the user is there. Reacts to
 * document and layout changes, since virtualized rendering defeats DOM-mutation watching.
 */
const stickToBottomPlugin = ViewPlugin.fromClass(
  class {
    pinned = true;

    constructor(private readonly view: EditorView) {
      view.scrollDOM.addEventListener('scroll', this.handleScroll);
      this.scrollToBottom();
    }

    update(update: ViewUpdate) {
      const moved = update.docChanged || update.geometryChanged;
      if (
        update.transactions.some((tr) => tr.effects.some((effect) => effect.is(pinToBottom))) ||
        (moved && this.isScrollLocked())
      ) {
        this.pinned = true;
      }
      // Wrapping, resizes and measured line heights all move the bottom too.
      if (this.pinned && moved) {
        this.scrollToBottom();
      }
    }

    destroy() {
      this.view.scrollDOM.removeEventListener('scroll', this.handleScroll);
    }

    private handleScroll = () => {
      // Only CodeMirror (e.g. search) can scroll a locked tail; stay pinned so the next
      // update brings the newest lines back.
      if (this.isScrollLocked()) return;
      const { scrollHeight, clientHeight, scrollTop } = this.view.scrollDOM;
      this.pinned = Math.abs(scrollHeight - clientHeight - scrollTop) < AT_BOTTOM_THRESHOLD_PX;
    };

    // LogViewer locks the unwrapped tail with overflow-y: hidden (styles.css).
    private isScrollLocked() {
      return getComputedStyle(this.view.scrollDOM).overflowY === 'hidden';
    }

    private scrollToBottom() {
      // Scroll in CodeMirror's write phase, after it has laid out the new content.
      this.view.requestMeasure({
        key: this,
        read: () => null,
        write: (_, view) => {
          view.scrollDOM.scrollTop = view.scrollDOM.scrollHeight;
        },
      });
    }
  }
);

export const stickToBottom: Extension = stickToBottomPlugin;

export const isPinnedToBottom = (view: EditorView): boolean =>
  view.plugin(stickToBottomPlugin)?.pinned ?? false;
