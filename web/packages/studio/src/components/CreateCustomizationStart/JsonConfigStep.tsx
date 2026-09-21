// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Block, Button, Flex, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { JsonConfigPanel } from '@studio/components/CreateCustomizationStart/JsonConfigPanel';
import type { CustomizationFormFields } from '@studio/util/forms/customization';
import { ArrowLeft } from 'lucide-react';
import { useState, type FC } from 'react';

interface Props {
  onBack: () => void;
  onContinue: (fields: CustomizationFormFields) => void;
}

/**
 * The second screen of the "start from a JSON config" path. Kept apart from the options
 * page rather than folded into it: the config is a whole screen's worth of editing, and
 * reaching it is a step the user can go back from.
 */
export const JsonConfigStep: FC<Props> = ({ onBack, onContinue }) => {
  const [fields, setFields] = useState<CustomizationFormFields | null>(null);

  return (
    <Stack className="h-full">
      <Block className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <Stack gap="density-2xl" padding="density-2xl" className="min-h-0 flex-1">
          <PageHeader
            slotHeading="Load a job config"
            slotDescription="Paste the config or upload the file. It is checked as you type, and opens in the form once it reads."
          />
          <Flex justify="center" className="min-h-0 w-full flex-1">
            <div className="flex min-h-0 w-full max-w-[768px] flex-1 flex-col">
              <JsonConfigPanel onValidConfig={setFields} />
            </div>
          </Flex>
        </Stack>
      </Block>

      <Flex justify="center" className="shrink-0 border-t border-base bg-surface-base px-10 py-3">
        <Flex align="center" justify="between" className="w-full max-w-[768px]">
          <Button kind="tertiary" onClick={onBack}>
            <ArrowLeft size={16} aria-hidden />
            Back
          </Button>
          <Button
            color="brand"
            kind="primary"
            disabled={fields === null}
            onClick={() => fields && onContinue(fields)}
          >
            Continue
          </Button>
        </Flex>
      </Flex>
    </Stack>
  );
};
