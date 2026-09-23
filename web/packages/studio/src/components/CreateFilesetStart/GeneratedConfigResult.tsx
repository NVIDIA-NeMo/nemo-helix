// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner, Button, Flex, Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import { GeneratedConfigPanel } from '@studio/components/CreateFilesetStart/GeneratedConfigPanel';
import type { GeneratedConfigResultProps } from '@studio/components/CreateFilesetStart/types';
import { FileJson, Sparkles, Wand2 } from 'lucide-react';
import { type FC, useState } from 'react';

const IssueList: FC<{ items: string[] }> = ({ items }) => (
  <ul className="list-disc pl-density-lg">
    {items.map((item) => (
      <li key={item}>
        <Text kind="body/regular/sm">{item}</Text>
      </li>
    ))}
  </ul>
);

/** Hands the listed issues back to the model. Rendered inside the banner it belongs to. */
const FixButton: FC<{ label: string; onFix: () => void }> = ({ label, onFix }) => (
  <Flex justify="start" className="mt-density-sm">
    <Button kind="secondary" size="small" onClick={onFix}>
      <Wand2 size={14} aria-hidden />
      {label}
    </Button>
  </Flex>
);

/**
 * Verdict on the last generated draft: whether it can be loaded into the build canvas, plus
 * what would be lost or had to be substituted. Rendered next to the prompt so a failed draft
 * is read as "the model got it wrong, refine and regenerate" rather than a broken page.
 */
export const GeneratedConfigResult: FC<GeneratedConfigResultProps> = ({
  validation,
  requestError,
  rawOutput,
  isGenerating,
  isFixing,
  onFix,
}) => {
  const [isConfigOpen, setIsConfigOpen] = useState(false);

  if (isGenerating || isFixing) {
    return (
      <Flex
        align="center"
        justify="center"
        gap="density-sm"
        className="h-full min-h-[220px] rounded-md border border-base bg-surface-raised p-density-xl"
      >
        <Spinner size="small" aria-label={isFixing ? 'Fixing' : 'Generating'} />
        <Text kind="body/regular/sm" className="text-secondary">
          {isFixing ? 'Working through the issues…' : 'Drafting your columns…'}
        </Text>
      </Flex>
    );
  }

  if (requestError) {
    return (
      <Banner kind="inline" status="error">
        {requestError}
      </Banner>
    );
  }

  if (!validation) {
    return (
      <Stack
        gap="density-xs"
        align="center"
        className="h-full min-h-[220px] justify-center rounded-md border border-dashed border-base p-density-xl text-center"
      >
        <Sparkles size={18} className="text-secondary" aria-hidden />
        <Text kind="body/semibold/sm" className="text-primary">
          No draft yet
        </Text>
        <Text kind="body/regular/sm" className="text-secondary">
          Describe the fileset you need, then generate. We check the result before it can be loaded
          into the builder.
        </Text>
      </Stack>
    );
  }

  const fixHandler = rawOutput !== null ? onFix : undefined;

  return (
    <Stack gap="density-md">
      {rawOutput ? (
        <>
          <Flex justify="end">
            <Button kind="tertiary" size="small" onClick={() => setIsConfigOpen(true)}>
              <FileJson size={14} aria-hidden />
              View config
            </Button>
          </Flex>
          <GeneratedConfigPanel
            open={isConfigOpen}
            config={rawOutput}
            onClose={() => setIsConfigOpen(false)}
          />
        </>
      ) : null}

      {validation.status === 'valid' ? (
        <>
          <Banner kind="inline" status="success">
            Valid job config — ready to load into the builder.
          </Banner>
          <Stack gap="density-xs" className="rounded-md border border-base p-density-lg">
            <Text kind="label/bold/sm" className="text-secondary">
              {validation.seed.name} · {validation.seed.rows} records ·{' '}
              {validation.seed.columns.length} column(s) · {validation.seed.models.length} model(s)
            </Text>
            <Stack gap="density-xxs">
              {validation.seed.columns.map((column) => (
                <Flex key={column.id} align="center" justify="between" gap="density-sm">
                  <Text kind="body/regular/sm" className="text-primary">
                    {column.name}
                  </Text>
                  <Text kind="body/regular/xs" className="text-secondary">
                    {column.option.label}
                  </Text>
                </Flex>
              ))}
            </Stack>
          </Stack>
        </>
      ) : (
        <Banner kind="inline" status="error">
          This draft can&apos;t be loaded into the builder yet:
          <IssueList items={validation.errors} />
          {fixHandler ? <FixButton label="Fix these errors" onFix={fixHandler} /> : null}
        </Banner>
      )}

      {validation.warnings.length > 0 && (
        <Banner kind="inline" status="warning">
          <IssueList items={validation.warnings} />
          {fixHandler && validation.status === 'valid' ? (
            <FixButton label="Fix these warnings" onFix={fixHandler} />
          ) : null}
        </Banner>
      )}
    </Stack>
  );
};
