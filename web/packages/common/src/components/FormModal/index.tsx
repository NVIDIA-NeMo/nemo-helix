/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import {
  Block,
  Button,
  Flex,
  ModalContent,
  ModalDialog,
  ModalFooter,
  ModalHeading,
  ModalMain,
  ModalRoot,
  Stack,
} from '@nvidia/foundations-react-core';
import cn from 'classnames';
import {
  ComponentProps,
  ComponentPropsWithoutRef,
  FC,
  PropsWithChildren,
  ReactNode,
  useId,
} from 'react';

export interface FormModalProps {
  open: boolean;
  title: ReactNode;
  instruction?: string;
  submitButtonText: string;
  errorText?: string | null;
  cancelButtonText?: string;
  /**
   * Disables everything, the submit button, the cancel button, and the ability to close the modal at all.
   * Use this to prevent the user from editing/closing/submitting the modal while request is in flight.
   */
  disabled?: boolean;
  loading?: boolean;
  /**
   * Only disables the submit button.
   * Use this to prevent the user from submitting until the form is valid and dirty.
   * */
  submitDisabled?: boolean;
  onSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
  onClose: () => void;
  styles?: React.CSSProperties;
  className?: string;
  slotAboveBody?: ReactNode;
  slotFooterLeft?: ReactNode;
  slotFooterRight?: ReactNode;
  attributes?: {
    CancelButton?: ComponentProps<typeof Button>;
    SubmitButton?: ComponentProps<typeof LoadingButton>;
    Form?: ComponentPropsWithoutRef<'form'>;
  };
}

export const FormModal: FC<PropsWithChildren<FormModalProps>> = ({
  open,
  title,
  instruction,
  submitButtonText,
  cancelButtonText,
  errorText = null,
  disabled = false,
  loading = false,
  submitDisabled = false,
  onSubmit,
  onClose,
  children,
  className,
  slotAboveBody,
  slotFooterLeft,
  slotFooterRight,
  attributes,
}) => {
  const modalId = useId();
  // Prevents the user from closing the dialog if the inputs are disabled
  const handleUserClose = () => {
    if (!disabled) {
      onClose();
    }
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.stopPropagation(); // Prevent event from bubbling to parent forms
    onSubmit(e);
  };

  const footerJustifyClass = slotFooterLeft ? 'justify-between' : 'justify-end';

  return (
    <ModalRoot id={modalId} open={open} onOpenChange={handleUserClose}>
      <ModalDialog>
        <ModalContent className={`max-h-[90vh] ${className || ''}`}>
          <form className="contents" onSubmit={handleSubmit} noValidate {...attributes?.Form}>
            <ModalHeading>{title}</ModalHeading>
            {slotAboveBody ? <Block className="shrink-0">{slotAboveBody}</Block> : null}
            <ModalMain className="flex-1 min-h-0 overflow-y-auto">
              <Stack gap="density-md" className="pt-4">
                {errorText && <p className="text-feedback-danger whitespace-normal">{errorText}</p>}
                {instruction && <p className="whitespace-normal">{instruction}</p>}
                {children}
              </Stack>
            </ModalMain>
            <ModalFooter
              className={cn('flex w-full gap-2 flex-1 flex-shrink-0', footerJustifyClass)}
            >
              {slotFooterLeft}
              {slotFooterRight ?? (
                <Flex gap="2">
                  <Button
                    kind="tertiary"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleUserClose();
                    }}
                    disabled={disabled}
                    type="button"
                    {...attributes?.CancelButton}
                  >
                    {cancelButtonText || 'Cancel'}
                  </Button>
                  <LoadingButton
                    type="submit"
                    disabled={disabled || submitDisabled}
                    onClick={(e) => {
                      e.stopPropagation();
                    }}
                    loading={loading}
                    {...attributes?.SubmitButton}
                  >
                    {submitButtonText}
                  </LoadingButton>
                </Flex>
              )}
            </ModalFooter>
          </form>
        </ModalContent>
      </ModalDialog>
    </ModalRoot>
  );
};
