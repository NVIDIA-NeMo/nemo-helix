// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CodeEditor } from '@nemo/common/src/components/CodeEditor';
import { ContentType } from '@nemo/common/src/components/CodeEditor/constants';
import { Banner, Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { parseCustomizationJson } from '@studio/components/CreateCustomizationStart/jsonConfig';
import type { JsonConfigPanelProps } from '@studio/components/CreateCustomizationStart/types';
import { CustomizationBackend } from '@studio/util/customizationBackend';
import { FileCode, Upload } from 'lucide-react';
import { useRef, useState, type ChangeEvent, type FC } from 'react';

/** How each detected backend is named back to the user in the success banner. */
const BACKEND_LABEL: Record<CustomizationBackend, string> = {
  [CustomizationBackend.automodel]: 'Automodel',
  [CustomizationBackend.unsloth]: 'Unsloth',
  [CustomizationBackend.rl]: 'RL',
};

/**
 * Offered behind a button rather than used as placeholder text: the editor has no
 * placeholder affordance, and seeding it directly would leave the panel looking like it
 * held a real config — one that parses, so Continue would light up on values nobody typed.
 */
const EXAMPLE = `{
  "spec": {
    "model": "default/qwen3-0-6b",
    "dataset": { "training": "default/hellaswag-demo" },
    "training": { "finetuning_type": "lora", "lora": { "rank": 64 } }
  }
}`;

export const JsonConfigPanel: FC<JsonConfigPanelProps> = ({ onValidConfig }) => {
  const [text, setText] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [backendLabel, setBackendLabel] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validate = (value: string) => {
    setText(value);
    // An empty editor is the resting state, not an error to shout about.
    if (!value.trim()) {
      setError(null);
      setBackendLabel(null);
      onValidConfig(null);
      return;
    }
    const result = parseCustomizationJson(value);
    if (result.ok) {
      setError(null);
      setBackendLabel(BACKEND_LABEL[result.backend]);
      onValidConfig(result.fields);
    } else {
      setError(result.error);
      setBackendLabel(null);
      onValidConfig(null);
    }
  };

  const handleFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    // Reset first so re-picking the same file still fires a change event.
    event.target.value = '';
    if (!file) return;
    validate(await file.text());
  };

  return (
    <Stack gap="density-md" className="w-full">
      <Flex align="center" justify="between" className="w-full gap-4">
        <Text kind="body/regular/sm" className="text-secondary">
          Paste a full job request (<code>{'{ "spec": … }'}</code>) or just the spec. Unspecified
          fields use the platform default, and you can edit everything on the next screen.
        </Text>
        <Flex align="center" gap="density-sm" className="shrink-0">
          <Button kind="tertiary" onClick={() => validate(EXAMPLE)}>
            <FileCode size={16} aria-hidden />
            Insert example
          </Button>
          <Button kind="secondary" onClick={() => fileInputRef.current?.click()}>
            <Upload size={16} aria-hidden />
            Upload .json
          </Button>
        </Flex>
        <input
          ref={fileInputRef}
          type="file"
          accept="application/json,.json"
          className="hidden"
          aria-label="Upload a JSON job config"
          onChange={(event) => void handleFile(event)}
        />
      </Flex>

      <div className="h-[360px] overflow-auto rounded-md border border-base">
        <CodeEditor
          id="job-config-json"
          className="h-full"
          content={text}
          contentType={ContentType.JSON}
          onChange={validate}
          hideCopyButton
        />
      </div>

      {error ? (
        <Banner kind="inline" status="error">
          {error}
        </Banner>
      ) : null}

      {backendLabel ? (
        <Banner kind="inline" status="success">
          {`Recognised a ${backendLabel} job. Continue to review it in the form.`}
        </Banner>
      ) : null}
    </Stack>
  );
};
