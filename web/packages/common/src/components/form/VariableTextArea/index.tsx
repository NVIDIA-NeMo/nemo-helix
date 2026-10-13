// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EditorView } from '@codemirror/view';
import {
  knownVariablesField,
  setKnownVariables,
  setVariableMatcher,
  textareaLook,
  tooltipWithinEditor,
  variableCompletions,
  variableDecorations,
  variableMatcherField,
  type VariableMatcher,
  variablesCompartment,
} from '@nemo/common/src/components/form/VariableTextArea/extensions';
import CodeMirror from '@uiw/react-codemirror';
import cn from 'classnames';
import {
  type ComponentProps,
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
} from 'react';

import '@nemo/common/src/components/form/VariableTextArea/styles.css';

export interface VariableDef {
  name: string;
  description?: string;
}

export interface VariableTextAreaHandle {
  insertVariable: (name: string) => void;
  focus: () => void;
}

export interface VariableTextAreaProps {
  value: string;
  onChange: (next: string) => void;
  variables?: VariableDef[];
  /** Overrides the known/unknown decision for a token. Defaults to exact
   *  membership of `variables`. */
  isKnown?: VariableMatcher;
  placeholder?: string;
  disabled?: boolean;
  readOnly?: boolean;
  autoFocus?: boolean;
  minRows?: number;
  maxRows?: number;
  className?: string;
  onFocus?: () => void;
  onBlur?: () => void;
  /**
   * Slot props forwarded onto the wrapper div (`Root`) and the CodeMirror
   * contentDOM (`TextAreaElement`). The `Record<`data-${string}`, string>`
   * intersection is intentional — `ComponentProps<'div'>` doesn't permit
   * arbitrary `data-*` keys, but consumers commonly pass `data-testid`.
   */
  attributes?: {
    Root?: ComponentProps<'div'> & Record<`data-${string}`, string>;
    TextAreaElement?: ComponentProps<'div'> & Record<`data-${string}`, string>;
  };
}

export const VariableTextArea = forwardRef<VariableTextAreaHandle, VariableTextAreaProps>(
  (
    {
      value,
      onChange,
      variables,
      isKnown,
      placeholder,
      disabled,
      readOnly,
      autoFocus,
      onFocus,
      onBlur,
      className,
      attributes,
    },
    ref
  ) => {
    const viewRef = useRef<EditorView | null>(null);
    const didInitialDispatch = useRef(false);

    const variableNames = useMemo(
      () => (variables ?? []).map((v) => v.name),
      // eslint-disable-next-line react-hooks/exhaustive-deps
      [JSON.stringify((variables ?? []).map((v) => v.name))]
    );

    const variablesRef = useRef<VariableDef[]>(variables ?? []);
    useEffect(() => {
      variablesRef.current = variables ?? [];
    }, [variables]);

    const completionsExtension = useMemo(
      () => variableCompletions({ getVariables: () => variablesRef.current }),
      []
    );

    useEffect(() => {
      if (!didInitialDispatch.current) {
        didInitialDispatch.current = true;
        return;
      }
      const view = viewRef.current;
      if (!view) return;
      const next = new Set(variableNames);
      view.dispatch({ effects: setKnownVariables.of(next) });
      view.dom.setAttribute('data-known-variables', JSON.stringify([...next]));
    }, [variableNames]);

    useEffect(() => {
      viewRef.current?.dispatch({ effects: setVariableMatcher.of(isKnown ?? null) });
    }, [isKnown]);

    useImperativeHandle(
      ref,
      () => ({
        insertVariable: (name: string) => {
          const view = viewRef.current;
          if (!view) return;
          const { from, to } = view.state.selection.main;
          const insert = `{{${name}}}`;
          view.dispatch({
            changes: { from, to, insert },
            selection: { anchor: from + insert.length },
          });
          view.focus();
        },
        focus: () => viewRef.current?.focus(),
      }),
      []
    );

    return (
      <div className={cn('nv-variable-text-area', className)} {...attributes?.Root}>
        <CodeMirror
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          editable={!disabled}
          readOnly={readOnly}
          autoFocus={autoFocus}
          basicSetup={false}
          onCreateEditor={(view) => {
            viewRef.current = view;
            view.contentDOM.setAttribute('role', 'textbox');
            const initial = new Set(variableNames);
            view.dispatch({
              effects: [setKnownVariables.of(initial), setVariableMatcher.of(isKnown ?? null)],
            });
            view.dom.setAttribute('data-known-variables', JSON.stringify([...initial]));
            const attrs = attributes?.TextAreaElement;
            if (attrs) {
              for (const [key, val] of Object.entries(attrs)) {
                if (val == null) continue;
                if (typeof val === 'function') continue;
                view.contentDOM.setAttribute(key, String(val));
              }
            }
          }}
          extensions={[
            textareaLook,
            variablesCompartment.of([knownVariablesField, variableMatcherField]),
            variableDecorations,
            tooltipWithinEditor,
            completionsExtension,
          ]}
          onFocus={onFocus}
          onBlur={onBlur}
        />
      </div>
    );
  }
);

VariableTextArea.displayName = 'VariableTextArea';
