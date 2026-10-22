// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { RailsConfig } from '@nemo/sdk/generated/platform/schema';
import {
  Button,
  Divider,
  Flex,
  SidePanel,
  Stack,
  Text,
  Tooltip,
} from '@nvidia/foundations-react-core';
import { Cog } from '@nvidia/react-gui-icons/icons';
import { PromptScopeSection } from '@studio/routes/guardrails/rails/components/PromptScopeSection';
import {
  findPrompt,
  withPromptContent,
  withPromptFields,
} from '@studio/routes/guardrails/rails/configOps';
import {
  SELF_CHECK_SCOPE_ORDER,
  SELF_CHECK_SCOPES,
  isSelfCheckScopeEnabled,
  setSelfCheckScopeEnabled,
} from '@studio/routes/guardrails/rails/selfCheck/bindings';
import type { RailSettingsProps } from '@studio/routes/guardrails/rails/types';
import { Fragment, type FC, useState } from 'react';

/**
 * Cog for the self-check rail: its trigger, its panel, and its body.
 *
 * The rail owns all three. There is no shared settings shell — what a rail needs to
 * configure varies too much for one to fit, and a rail with simpler needs can open a
 * popover, expand inline, or offer nothing at all.
 *
 * No model picker here: self check runs on whichever model is already serving the
 * request, unlike content safety which binds a task LLM.
 */
export const SelfCheckSettings: FC<RailSettingsProps> = ({ data, onChange }) => {
  const [open, setOpen] = useState(false);
  // Edits land in a draft so the panel's two exits differ: Apply commits it to the
  // working copy, closing discards it.
  const [draft, setDraft] = useState<RailsConfig>(data);

  const openPanel = () => {
    setDraft(data); // always branch from the current working copy, not a stale one
    setOpen(true);
  };

  return (
    <>
      <Tooltip slotContent="Configure Self Checks">
        <Button
          kind="tertiary"
          color="neutral"
          onClick={openPanel}
          aria-label="Configure Self Checks"
        >
          <Cog size="16" />
        </Button>
      </Tooltip>

      <SidePanel
        className="w-[min(560px,90vw)]"
        bordered
        // Modal, like every Studio side panel that carries a footer. A non-modal panel
        // sits in normal page stacking, so page-level fixed elements float over its
        // actions; the overlay puts them above.
        modal
        open={open}
        onOpenChange={(next) => {
          if (!next) setOpen(false);
        }}
        slotHeading={<Text kind="label/bold/lg">Self Checks Rail</Text>}
        slotFooter={
          <Flex justify="end" gap="density-lg" className="w-full">
            <Button kind="tertiary" type="button" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            {/*
              "Apply", not "Save": this only updates the guardrail's working copy. Nothing
              reaches the server until the page's Save Guardrail.
            */}
            <Button
              color="brand"
              type="button"
              onClick={() => {
                onChange(draft);
                setOpen(false);
              }}
            >
              Apply
            </Button>
          </Flex>
        }
      >
        <Stack gap="density-xl">
          <Text kind="body/regular/sm" className="text-text-secondary">
            Asks the model answering the request to judge whether a message should be blocked. Needs
            no separate safety model.
          </Text>

          {SELF_CHECK_SCOPE_ORDER.map((scope, index) => {
            const binding = SELF_CHECK_SCOPES[scope];
            return (
              <Fragment key={scope}>
                {index > 0 ? <Divider /> : null}
                <PromptScopeSection
                  scope={scope}
                  enabled={isSelfCheckScopeEnabled(draft, scope)}
                  onEnabledChange={(enabled) =>
                    setDraft(setSelfCheckScopeEnabled(draft, scope, enabled))
                  }
                  prompt={findPrompt(draft, binding.task)?.content ?? binding.defaultPrompt}
                  onPromptChange={(content) =>
                    setDraft(withPromptContent(draft, binding.task, content))
                  }
                  variables={binding.variables}
                  maxTokens={findPrompt(draft, binding.task)?.max_tokens}
                  onMaxTokensChange={(max_tokens) =>
                    setDraft(withPromptFields(draft, binding.task, { max_tokens }))
                  }
                  maxLength={findPrompt(draft, binding.task)?.max_length}
                  onMaxLengthChange={(max_length) =>
                    setDraft(withPromptFields(draft, binding.task, { max_length }))
                  }
                  stop={findPrompt(draft, binding.task)?.stop ?? []}
                  onStopChange={(stop) =>
                    setDraft(
                      withPromptFields(draft, binding.task, {
                        stop: stop.length ? stop : undefined,
                      })
                    )
                  }
                />
              </Fragment>
            );
          })}
        </Stack>
      </SidePanel>
    </>
  );
};
