// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  AccordionContent,
  AccordionItem,
  AccordionRoot,
  AccordionTrigger,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { SOURCE_TABS, TAB_LABELS } from '@studio/components/CreateDatasetPanel/constants';
import type { Draft, PanelAction } from '@studio/components/CreateDatasetPanel/draft';
import {
  DescriptionField,
  NameField,
  SecretSelect,
  TextField,
} from '@studio/components/CreateDatasetPanel/fields';
import type { SourceTab } from '@studio/components/CreateDatasetPanel/parseSource';
import type { WorkspaceSecrets } from '@studio/components/CreateDatasetPanel/useWorkspaceSecrets';
import type { DraftErrors, SourceError } from '@studio/components/CreateDatasetPanel/validateDraft';
import { type Dispatch, type ReactNode, useRef, useState } from 'react';

interface SourceTabFormProps {
  tab: SourceTab;
  draft: Draft;
  /** Only the errors that should currently be on screen. */
  errors: DraftErrors;
  secrets: WorkspaceSecrets;
  dispatch: Dispatch<PanelAction>;
}

function renderSourceError(
  error: SourceError | undefined,
  dispatch: Dispatch<PanelAction>
): ReactNode {
  if (!error) return undefined;
  if (error.kind === 'message') return error.message;
  // Switching tabs does not carry the URL over; the other tab keeps its own draft (spec 4.2).
  const label = TAB_LABELS[error.actual];
  return (
    <>
      Trying to import from {label}? Switch to{' '}
      <button
        type="button"
        className="underline"
        onClick={() => dispatch({ type: 'tab-selected', tab: error.actual })}
      >
        {label}
      </button>{' '}
      tab.
      {/* The links above split the sentence across elements; this unbroken copy keeps it findable as
          text. Assistive tech reads the live message above, so the copy is hidden from it. */}
      <span className="sr-only" aria-hidden="true">
        Trying to import from {label}? Switch to {label} tab.
      </span>
    </>
  );
}

/** HuggingFace, NGC, GitHub and S3 tabs: source URL, name, source settings, description, advanced. */
export function SourceTabForm({ tab, draft, errors, secrets, dispatch }: SourceTabFormProps) {
  const config = SOURCE_TABS[tab];
  // A paste event arrives just before the change event that carries the pasted text.
  const pasted = useRef(false);
  const [advancedOpen, setAdvancedOpen] = useState('');

  return (
    <Stack gap="density-lg" className="w-full min-w-0">
      <TextField
        label={config.urlLabel}
        value={draft.source}
        placeholder={config.urlPlaceholder}
        error={renderSourceError(errors.source, dispatch)}
        onPaste={() => {
          pasted.current = true;
        }}
        onChange={(value) => {
          dispatch({ type: 'source-changed', value, pasted: pasted.current });
          pasted.current = false;
        }}
        onBlur={() => dispatch({ type: 'source-blurred' })}
      />
      <NameField value={draft.name} error={errors.name} dispatch={dispatch} />
      {config.basic.map((field) =>
        field.kind === 'secret' ? (
          <SecretSelect
            key={field.key}
            label={field.label}
            value={draft.values[field.key] ?? ''}
            required={Boolean(field.requiredMessage)}
            error={errors.secrets[field.key]}
            secrets={secrets}
            onChange={(value) => dispatch({ type: 'value-changed', key: field.key, value })}
          />
        ) : (
          <TextField
            key={field.key}
            label={field.label}
            value={draft.values[field.key] ?? ''}
            placeholder={field.placeholder}
            onChange={(value) => dispatch({ type: 'value-changed', key: field.key, value })}
          />
        )
      )}
      <DescriptionField value={draft.description} dispatch={dispatch} />
      {/* Collapsed content is not mounted: its values live in the draft and still reach the request. */}
      <AccordionRoot collapsible value={advancedOpen} onValueChange={setAdvancedOpen}>
        <AccordionItem value="advanced" className="border-b-0">
          <AccordionTrigger>
            <Text kind="label/bold/md">Advanced</Text>
          </AccordionTrigger>
          <AccordionContent>
            {advancedOpen === 'advanced' && (
              <Stack gap="density-lg" className="pt-density-md">
                {config.advanced.map((field) => (
                  <TextField
                    key={field.key}
                    label={field.label}
                    value={draft.values[field.key] ?? ''}
                    placeholder={field.placeholder}
                    onChange={(value) => dispatch({ type: 'value-changed', key: field.key, value })}
                  />
                ))}
              </Stack>
            )}
          </AccordionContent>
        </AccordionItem>
      </AccordionRoot>
    </Stack>
  );
}
