// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Toast } from '@nvidia/foundations-react-core';
import type { ComponentProps, ReactNode } from 'react';

export interface MessageFnOptions {
  durationMs?: number | false; // How long should the toast appear for? Pass false to prevent auto-dismiss.
}
interface AddToastFnOptions extends MessageFnOptions {
  status?: ComponentProps<typeof Toast>['status'];
}

export type MessageFn = (message: ReactNode, options?: MessageFnOptions) => void;
export type AddToastFn = (message: ReactNode, options: AddToastFnOptions) => string;
export type DismissToastFn = (id: string) => void;

export interface ToastObject {
  success: MessageFn;
  error: MessageFn;
  info: MessageFn;
  warning: MessageFn;
  working: MessageFn;
  /** Returns toast id so caller can dismiss it when done (e.g. to replace with success/error) */
  workingWithId: (message: ReactNode, options?: AddToastFnOptions) => string;
  neutral: MessageFn;
  dismissToast: DismissToastFn;
}

export type NotifyType = 'success' | 'error' | 'info' | 'warning';

/**
 * Provider-independent notification sink. Components exposed across the plugin
 * boundary take this as a prop so a plugin can route them at Studio's toaster
 * without sharing a ToastContext module instance.
 */
export type NotifyFn = (message: string, type?: NotifyType, options?: MessageFnOptions) => void;

export interface ToastContextValue {
  addToast: AddToastFn;
  dismissToast: DismissToastFn;
  toast: ToastObject;
}

export interface ToastDescriptor {
  id: string;
  message: ReactNode;
  status: ComponentProps<typeof Toast>['status'];
  isVisible?: boolean; // Handles opacity transition
}
