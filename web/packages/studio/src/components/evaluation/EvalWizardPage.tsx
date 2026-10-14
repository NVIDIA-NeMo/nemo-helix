// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import {
  Block,
  Button,
  Flex,
  PageHeader,
  Panel,
  Stack,
  Stepper,
  Text,
} from '@nvidia/foundations-react-core';
import '@studio/components/evaluation/evalWizard.css';
import { CONTENT_WIDTH } from '@studio/components/StartOptions/tile';
import type { FC, FormEvent, ReactNode } from 'react';
import { useLocation, useNavigate } from 'react-router';

interface EvalWizardPageProps {
  heading: string;
  headingDescription: string;
  /** Step headings, in order. */
  steps: string[];
  activeStep: number;
  /** Where Cancel lands when there is no in-app page to go back to, e.g. after a deep link. */
  cancelTo: string;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  /** Rendered under the stepper, where it stays in view — a submit error, typically. */
  slotBanner?: ReactNode;
  /** Rendered at the start of the actions row — the step's Back button. */
  slotFooterStart?: ReactNode;
  continueLabel: ReactNode;
  continueLoading?: boolean;
  canContinue: boolean;
  onContinue: () => void;
  /** Says what is still missing, next to a disabled Continue. */
  blockedHint?: string;
  children: ReactNode;
}

/**
 * A new-evaluation wizard as a page: the header and stepper hold their place, and the current
 * step scrolls under them in a panel whose footer carries Back, Cancel and Continue.
 */
export const EvalWizardPage: FC<EvalWizardPageProps> = ({
  heading,
  headingDescription,
  steps,
  activeStep,
  cancelTo,
  onSubmit,
  slotBanner,
  slotFooterStart,
  continueLabel,
  continueLoading = false,
  canContinue,
  onContinue,
  blockedHint,
  children,
}) => {
  const navigate = useNavigate();
  const location = useLocation();
  // Back to wherever the wizard was opened from, so a cancelled form is not left in history.
  const cancel = () =>
    location.key === 'default' ? navigate(cancelTo, { replace: true }) : navigate(-1);

  return (
    <AccessibleTitle title={heading}>
      <form noValidate onSubmit={onSubmit} className="h-full">
        <Stack className="h-full">
          <Stack gap="density-2xl" className="shrink-0 px-6 pb-4 pt-6">
            <PageHeader slotHeading={heading} slotDescription={headingDescription} />

            <Flex justify="center" className="w-full">
              <Stack gap="density-2xl" className={CONTENT_WIDTH}>
                <Stepper
                  className="eval-wizard-stepper"
                  aria-label={`${heading} progress`}
                  activeStep={activeStep}
                  items={steps.map((slotHeading) => ({ slotHeading }))}
                />
                {slotBanner}
              </Stack>
            </Flex>
          </Stack>

          <Block className="min-h-0 flex-1 overflow-auto px-6 pb-6 pt-4 [scrollbar-gutter:stable]">
            <Flex justify="center" className="w-full">
              <Panel
                elevation="high"
                density="standard"
                className={CONTENT_WIDTH}
                slotFooter={
                  <Flex align="center" justify="between" gap="density-md" className="w-full">
                    <Flex align="center" gap="density-md">
                      {slotFooterStart}
                    </Flex>
                    <Flex align="center" gap="density-md">
                      {!canContinue && blockedHint ? (
                        <Text kind="label/regular/md" className="text-secondary">
                          {blockedHint}
                        </Text>
                      ) : null}
                      <Button kind="tertiary" onClick={cancel} disabled={continueLoading}>
                        Cancel
                      </Button>
                      <LoadingButton
                        color="brand"
                        kind="primary"
                        loading={continueLoading}
                        onClick={onContinue}
                        disabled={!canContinue}
                      >
                        {continueLabel}
                      </LoadingButton>
                    </Flex>
                  </Flex>
                }
              >
                {children}
              </Panel>
            </Flex>
          </Block>
        </Stack>
      </form>
    </AccessibleTitle>
  );
};
