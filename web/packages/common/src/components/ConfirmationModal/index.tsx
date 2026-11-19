// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal, type FormModalProps } from '@nemo/common/src/components/FormModal';
import type { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import type { NotifyFn } from '@nemo/common/src/providers/toast/types';
import { useNotify } from '@nemo/common/src/providers/toast/useNotify';
import { handleFormErrorsGeneric } from '@nemo/common/src/utils/forms/error';
import { Stack, Text } from '@nvidia/foundations-react-core';
import { type ComponentProps, type FC, type ReactNode, useEffect, useRef, useState } from 'react';
import { useForm } from 'react-hook-form';

interface FormFields {
  confirmText: string;
}

type SubmitButtonColor = NonNullable<ComponentProps<typeof LoadingButton>['color']>;

export interface ConfirmationModalProps extends Pick<
  FormModalProps,
  'open' | 'onClose' | 'cancelButtonText' | 'initialFocus'
> {
  /** Return true when the action succeeded (shows success toast); false shows error toast. */
  onConfirm: () => boolean | Promise<boolean>;
  title: string;
  /** Body copy shown above the optional confirmation field. */
  description: string;
  /** When set with `simpleConfirm === false`, user must type this exact string to enable submit. */
  confirmationText?: string;
  /** If true (default), only a single button click is required. Set `false` to require typing `confirmationText`. */
  simpleConfirm?: boolean;
  successText?: string;
  errorText?: string;
  submitButtonText?: string;
  /** Passed to the submit button; omit for default (non-destructive) styling. */
  submitButtonColor?: SubmitButtonColor;
  /** When true, success and error toasts from this modal are skipped (caller handles feedback). */
  suppressResultToasts?: boolean;
  /** Where result messages go. Defaults to the surrounding ToastProvider; plugins pass `host.notifications.notify`. */
  onNotify?: NotifyFn;
  children?: ReactNode;
}

export const ConfirmationModal: FC<ConfirmationModalProps> = ({
  open,
  onClose,
  onConfirm,
  title,
  description,
  confirmationText,
  simpleConfirm = true,
  successText = 'Done.',
  errorText = 'Something went wrong. Please try again.',
  submitButtonText = 'Confirm',
  cancelButtonText,
  initialFocus,
  submitButtonColor,
  suppressResultToasts = false,
  onNotify,
  children,
}) => {
  const {
    reset,
    control,
    handleSubmit,
    formState: { isValid },
  } = useForm<FormFields>({
    mode: 'onChange',
    defaultValues: { confirmText: '' },
  });
  const [isPending, setIsPending] = useState(false);
  const activeSubmissionRef = useRef<object | null>(null);
  const notify = useNotify(onNotify);

  useEffect(() => {
    if (!open) {
      activeSubmissionRef.current = null;
      setIsPending(false);
    }
  }, [open]);

  const resetAndClose = () => {
    reset();
    onClose();
  };

  const abandonAndClose = () => {
    activeSubmissionRef.current = null;
    setIsPending(false);
    resetAndClose();
  };

  const onSubmit = async () => {
    const submission = {};
    activeSubmissionRef.current = submission;
    setIsPending(true);
    try {
      const ok = await onConfirm();
      if (!suppressResultToasts) {
        if (ok) {
          notify(successText, 'success');
        } else {
          notify(errorText, 'error');
        }
      }
    } catch (error: unknown) {
      // The API-aware `getErrorMessage`, not the one in `utils/error`: that one returns only
      // `error.message`, which for a rejected request is "Request failed with status code 409"
      // — it drops the `detail` the service sent explaining what to do about it. Non-Error
      // throws carry no message to read, so those still fall back to `errorText`.
      notify(error instanceof Error ? getErrorMessage(error, errorText) : errorText, 'error');
    } finally {
      if (activeSubmissionRef.current === submission) {
        activeSubmissionRef.current = null;
        resetAndClose();
        setIsPending(false);
      }
    }
  };

  const confirmationMessage = confirmationText ? `Type "${confirmationText}" to confirm` : '';

  const validateConfirmationText = (confirmationTextInput: string) =>
    confirmationTextInput === confirmationText || confirmationMessage;

  return (
    <FormModal
      open={open}
      title={title}
      submitButtonText={submitButtonText}
      cancelButtonText={cancelButtonText}
      initialFocus={initialFocus}
      disabled={isPending}
      cancelableWhileDisabled
      submitDisabled={simpleConfirm ? false : !isValid}
      loading={isPending}
      onSubmit={handleSubmit(
        onSubmit,
        handleFormErrorsGeneric({ title: 'Confirmation Form Errors' })
      )}
      onClose={abandonAndClose}
      attributes={
        submitButtonColor
          ? {
              SubmitButton: {
                color: submitButtonColor,
              },
            }
          : undefined
      }
    >
      <Stack gap="density-md" className={simpleConfirm ? 'mb-4' : undefined}>
        <Text className="leading-relaxed">{description}</Text>
        {children}
        {!simpleConfirm && (
          <ControlledTextInput
            useControllerProps={{
              control,
              name: 'confirmText',
              rules: {
                required: confirmationMessage,
                validate: (confirmationTextInput) =>
                  validateConfirmationText(confirmationTextInput),
              },
            }}
            label="Confirmation"
            required
            placeholder={confirmationMessage}
            autoFocus
          />
        )}
      </Stack>
    </FormModal>
  );
};
