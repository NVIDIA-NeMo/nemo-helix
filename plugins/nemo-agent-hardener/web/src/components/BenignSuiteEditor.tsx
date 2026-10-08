// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SuiteRow } from '@agent-hardener/components/hitlTypes';
import {
  Button,
  Flex,
  FormField,
  Stack,
  Text,
  TextArea,
  TextInput,
} from '@nvidia/foundations-react-core';
import { Plus, Trash } from 'lucide-react';
import { FC } from 'react';

interface BenignSuiteEditorProps {
  value: SuiteRow[];
  onChange: (rows: SuiteRow[]) => void;
  disabled?: boolean;
}

const EMPTY_ROW: SuiteRow = { tool: '', payload: '', label: 'benign', persona: '', rationale: '' };

// Structured editor for the manifest's cached benign suite. Each row is a replayed request
// (tool/payload/label/persona/rationale); add/remove/edit rows, then the parent persists via PATCH.
export const BenignSuiteEditor: FC<BenignSuiteEditorProps> = ({ value, onChange, disabled }) => {
  const update = (index: number, patch: Partial<SuiteRow>) =>
    onChange(value.map((row, i) => (i === index ? { ...row, ...patch } : row)));
  const remove = (index: number) => onChange(value.filter((_, i) => i !== index));
  const add = () => onChange([...value, { ...EMPTY_ROW }]);

  return (
    <Stack gap="density-lg">
      {value.length === 0 ? (
        <Text kind="body/regular/md" className="text-subtle">
          No benign requests yet. Add rows manually, or generate the suite to populate it.
        </Text>
      ) : (
        value.map((row, index) => (
          <Stack key={index} gap="density-sm" className="rounded-md border border-base p-3">
            <Flex gap="density-md">
              <FormField name={`tool-${index}`} slotLabel="Tool" className="flex-1">
                <TextInput
                  value={row.tool}
                  disabled={disabled}
                  onChange={(e) => update(index, { tool: e.target.value })}
                />
              </FormField>
              <FormField name={`persona-${index}`} slotLabel="Persona" className="flex-1">
                <TextInput
                  value={row.persona ?? ''}
                  disabled={disabled}
                  onChange={(e) => update(index, { persona: e.target.value })}
                />
              </FormField>
              <FormField name={`label-${index}`} slotLabel="Label" className="flex-1">
                <TextInput
                  value={row.label ?? ''}
                  disabled={disabled}
                  onChange={(e) => update(index, { label: e.target.value })}
                />
              </FormField>
            </Flex>
            <FormField name={`payload-${index}`} slotLabel="Payload">
              <TextArea
                value={row.payload}
                rows={2}
                disabled={disabled}
                onChange={(e) => update(index, { payload: e.target.value })}
              />
            </FormField>
            <FormField name={`rationale-${index}`} slotLabel="Rationale">
              <TextArea
                value={row.rationale ?? ''}
                rows={2}
                disabled={disabled}
                onChange={(e) => update(index, { rationale: e.target.value })}
              />
            </FormField>
            <Flex>
              <Button
                kind="tertiary"
                color="danger"
                disabled={disabled}
                onClick={() => remove(index)}
              >
                <Trash /> Remove
              </Button>
            </Flex>
          </Stack>
        ))
      )}
      <Flex>
        <Button kind="secondary" disabled={disabled} onClick={add}>
          <Plus /> Add request
        </Button>
      </Flex>
    </Stack>
  );
};
