// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Toast } from '@nvidia/foundations-react-core';
import { type ComponentProps, type FC, type PropsWithChildren, useState } from 'react';

import type { ToastContextValue } from '../providers/toast/types';
import { ToastContext } from '../providers/toast/useToast';

/**
 * Mock ToastProvider that eliminates timeouts and provides a clean testing interface.
 * This prevents timeout-related errors in tests while still providing the toast context.
 */
export const MockToastProvider: FC<PropsWithChildren> = ({ children }) => {
  const [toasts, setToasts] = useState<
    Array<{ id: string; message: React.ReactNode; status?: ComponentProps<typeof Toast>['status'] }>
  >([]);

  const dismissToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  const addToast: ToastContextValue['addToast'] = (message, options) => {
    const { status } = options;
    const newToastId = `mock-toast-${Date.now()}-${Math.random()}`;

    setToasts((prev) => [
      ...prev,
      {
        id: newToastId,
        message,
        status,
      },
    ]);

    return newToastId;
  };

  const contextValue: ToastContextValue = {
    addToast,
    dismissToast,
    toast: {
      success: (message, options) => addToast(message, { status: 'success', ...options }),
      error: (message, options) => addToast(message, { status: 'error', ...options }),
      info: (message, options) => addToast(message, { status: 'info', ...options }),
      warning: (message, options) => addToast(message, { status: 'warning', ...options }),
      working: (message, options) => addToast(message, { status: 'working', ...options }),
      workingWithId: (message, options) => addToast(message, { status: 'working', ...options }),
      neutral: (message, options) => addToast(message, { ...options }),
      dismissToast,
    },
  };

  return (
    <ToastContext.Provider value={contextValue}>
      {children}
      {/* Render toasts for visual testing if needed, but without timeouts */}
      <div
        data-testid="mock-toast-container"
        className="fixed top-0 right-4 flex flex-col items-end gap-4 z-1100 max-w-md"
      >
        {toasts.map((toast) => (
          <div
            key={toast.id}
            data-testid={`mock-toast-${toast.status || 'neutral'}`}
            data-toast-id={toast.id}
            className="opacity-100 transition-all duration-300"
          >
            {toast.message}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
};
