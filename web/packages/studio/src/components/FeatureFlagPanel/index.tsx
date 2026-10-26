// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flag } from '@nemo/icons/icons';
import {
  Badge,
  Button,
  ButtonGroup,
  Divider,
  Flex,
  Modal,
  ModalCloseButton,
  Select,
  Stack,
  Text,
  TextInput,
  Tooltip,
} from '@nvidia/foundations-react-core';
import { featureFlags, flagDefaults } from '@studio/constants/featureFlags';
import { flagDefinitions } from '@studio/constants/featureFlags/featureFlags';
import {
  NON_OVERRIDABLE_FLAGS,
  OverrideBlob,
  readStoredOverrides,
  writeStoredOverrides,
} from '@studio/constants/featureFlags/overrides';
import { PREVIEW } from '@studio/constants/featureFlags/utils';
import { useMemo, useState, type FC } from 'react';

/**
 * The gate flag is absent from the list because an override for it is discarded
 * on read, so a row would render a control that silently does nothing. Derived
 * from the storage layer's own rule so the two cannot disagree.
 */
const HIDDEN_FLAGS = NON_OVERRIDABLE_FLAGS;

/** Sentinel for "no entry in the blob", which is distinct from an explicit false. */
const USE_DEFAULT = '';

const BOOLEAN_CHOICES = ['true', 'false'];
const PREVIEW_CHOICES = ['true', PREVIEW, 'false'];

const isFixedChoice = (typeName: string) => typeName === 'boolean' || typeName === 'preview';

const formatValue = (value: unknown): string => {
  if (value === undefined || value === null) return '—';
  if (typeof value === 'string') return value === '' ? '(empty)' : value;
  return String(value);
};

const isValidValue = (typeName: string, value: string): boolean => {
  if (typeName === 'number') return value.trim() !== '' && Number.isFinite(Number(value));
  return true;
};

type FlagRow = {
  key: string;
  envVar: string;
  typeName: string;
  defaultValue: unknown;
  effectiveValue: unknown;
};

const rows: FlagRow[] = Object.entries(flagDefinitions)
  .filter(([key]) => !HIDDEN_FLAGS.has(key))
  .map(([key, descriptor]) => ({
    key,
    envVar: descriptor.envVar,
    typeName: descriptor.typeName,
    defaultValue: (flagDefaults as Record<string, unknown>)[key],
    effectiveValue: (featureFlags as Record<string, unknown>)[key],
  }))
  .sort((a, b) => a.key.localeCompare(b.key));

interface FlagRowProps {
  row: FlagRow;
  override: string | undefined;
  onChange: (key: string, value: string | undefined) => void;
}

const FeatureFlagRow: FC<FlagRowProps> = ({ row, override, onChange }) => {
  const isOverridden = override !== undefined;
  const invalid = isOverridden && !isValidValue(row.typeName, override);

  const detail = isOverridden
    ? `${row.typeName} · default ${formatValue(row.defaultValue)}`
    : `${row.typeName} · default ${formatValue(row.defaultValue)} · now ${formatValue(row.effectiveValue)}`;

  return (
    <Stack gap="density-xs" className={isOverridden ? 'border-l-2 border-brand pl-3' : 'pl-3'}>
      <Flex align="center" justify="between" gap="density-md">
        <Stack gap="density-xxs" className="min-w-0">
          <Tooltip slotContent={row.envVar} side="right">
            <Text kind="label/semibold/sm" className="font-mono break-all">
              {row.key}
            </Text>
          </Tooltip>
          <Text kind="body/regular/xs" className="text-secondary">
            {detail}
          </Text>
        </Stack>

        <Flex align="center" justify="end" gap="density-xs" className="w-64 shrink-0">
          {isFixedChoice(row.typeName) ? (
            <Select
              aria-label={`${row.key} override`}
              size="small"
              value={override ?? USE_DEFAULT}
              onValueChange={(value) =>
                onChange(row.key, value === USE_DEFAULT ? undefined : (value as string))
              }
              items={[
                { children: `Default (${formatValue(row.defaultValue)})`, value: USE_DEFAULT },
                ...(row.typeName === 'preview' ? PREVIEW_CHOICES : BOOLEAN_CHOICES).map(
                  (choice) => ({ children: choice, value: choice })
                ),
              ]}
            />
          ) : (
            <TextInput
              aria-label={`${row.key} override`}
              size="small"
              className="w-40"
              placeholder={formatValue(row.defaultValue)}
              value={override ?? ''}
              status={invalid ? 'error' : undefined}
              onChange={(event) => {
                const next = event.currentTarget.value;
                onChange(row.key, next === '' ? undefined : next);
              }}
            />
          )}
          {isOverridden && (
            <Tooltip slotContent="Clear override" side="left">
              <Button
                kind="tertiary"
                size="small"
                aria-label={`Clear ${row.key} override`}
                onClick={() => onChange(row.key, undefined)}
              >
                ✕
              </Button>
            </Tooltip>
          )}
        </Flex>
      </Flex>
    </Stack>
  );
};

/**
 * Local feature-flag override panel.
 *
 * Edits accumulate in a draft and commit as a single write plus one reload,
 * rather than reloading per change. Flags are parsed once at module load, so a
 * reload is the only way an override takes effect.
 */
export const FeatureFlagPanel: FC = () => {
  const stored = useMemo(() => readStoredOverrides(flagDefinitions), []);
  const [draft, setDraft] = useState<OverrideBlob>(stored);
  const [filter, setFilter] = useState('');
  const [saveFailed, setSaveFailed] = useState(false);

  const committed = useMemo(() => JSON.stringify(stored), [stored]);
  const isDirty = JSON.stringify(draft) !== committed;
  const count = Object.keys(draft).length;

  const hasInvalid = Object.entries(draft).some(([key, value]) => {
    const row = rows.find((r) => r.key === key);
    return row ? !isValidValue(row.typeName, value) : false;
  });

  const visible = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    if (!needle) return rows;
    return rows.filter(
      (row) => row.key.toLowerCase().includes(needle) || row.envVar.toLowerCase().includes(needle)
    );
  }, [filter]);

  const handleChange = (key: string, value: string | undefined) => {
    setSaveFailed(false);
    setDraft((previous) => {
      const next = { ...previous };
      if (value === undefined) delete next[key];
      else next[key] = value;
      return next;
    });
  };

  /**
   * Bulk-set every visible flag. Scoped to what the filter is showing, since
   * silently changing rows the user cannot see is the more surprising option.
   *
   * Only fixed-choice flags are touched: "true"/"false" is not a meaningful
   * value for a string or number flag, and the schema guard would drop it on
   * the next read anyway.
   *
   * Distinct from "Reset all", which removes overrides so flags fall back to
   * the deployment default — this writes an explicit value.
   */
  const setAllVisible = (value: 'true' | 'false') => {
    setSaveFailed(false);
    setDraft((previous) => {
      const next = { ...previous };
      for (const row of visible) {
        if (isFixedChoice(row.typeName)) next[row.key] = value;
      }
      return next;
    });
  };

  const bulkCount = visible.filter((row) => isFixedChoice(row.typeName)).length;

  const applyAndReload = () => {
    if (writeStoredOverrides(draft)) {
      window.location.reload();
    } else {
      setSaveFailed(true);
    }
  };

  return (
    <Modal
      // Wide enough that the flag name, its detail line, and the control all sit
      // on one row across 35 entries. KUI exposes no size prop; the fullscreen
      // example sets width through className the same way.
      className="w-[min(960px,92vw)]"
      // The draft is unsaved work, so a stray backdrop click must not discard it.
      closeOnClickOutside={false}
      slotHeading="Feature flags"
      // The trigger must be the Button itself: the dialog clones slotTrigger via
      // asChild, so wrapping it in a Tooltip hands the trigger props to the
      // tooltip and the click never reaches the dialog. aria-label carries the
      // accessible name instead.
      slotTrigger={
        <Button aria-label="Feature flags" color="neutral" kind="tertiary" size="medium">
          <Flag />
        </Button>
      }
      slotFooter={
        <>
          <ModalCloseButton kind="tertiary">Close</ModalCloseButton>
          <Flex align="center" gap="density-sm">
            {/* Grouped because all three act on the same set of rows; "Apply and
                reload" stays the one primary action on the trailing edge. */}
            <ButtonGroup kind="secondary">
              <Button disabled={bulkCount === 0} onClick={() => setAllVisible('true')}>
                All on
              </Button>
              <Button disabled={bulkCount === 0} onClick={() => setAllVisible('false')}>
                All off
              </Button>
              <Button disabled={count === 0} onClick={() => setDraft({})}>
                Reset all
              </Button>
            </ButtonGroup>
            <Button color="brand" disabled={!isDirty || hasInvalid} onClick={applyAndReload}>
              Apply and reload
            </Button>
          </Flex>
        </>
      }
    >
      <Stack gap="density-md">
        <Text kind="body/regular/sm" className="text-secondary">
          Overrides are stored in this browser only and take effect on reload. They do not affect
          the deployment or anyone else.
        </Text>

        <TextInput
          aria-label="Filter flags"
          type="search"
          placeholder="Filter flags…"
          value={filter}
          onChange={(event) => setFilter(event.currentTarget.value)}
        />

        <Flex align="center" gap="density-sm">
          <Badge color={isDirty ? 'blue' : 'gray'} kind="solid">
            {count} {count === 1 ? 'override' : 'overrides'} {isDirty ? 'pending' : 'active'}
          </Badge>
          {saveFailed && (
            <Text kind="body/regular/xs" className="text-feedback-danger">
              Could not save — browser storage rejected the write.
            </Text>
          )}
        </Flex>

        <Divider />

        {visible.length === 0 ? (
          <Text kind="body/regular/sm" className="text-secondary">
            No flags match “{filter}”.
          </Text>
        ) : (
          <Stack gap="density-md" className="max-h-[55vh] overflow-y-auto pr-2">
            {visible.map((row) => (
              <FeatureFlagRow
                key={row.key}
                row={row}
                override={draft[row.key]}
                onChange={handleChange}
              />
            ))}
          </Stack>
        )}
      </Stack>
    </Modal>
  );
};
