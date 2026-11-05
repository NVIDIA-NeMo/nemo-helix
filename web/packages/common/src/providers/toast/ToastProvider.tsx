// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  DEFAULT_TOAST_DISMISS_MS,
  ERROR_TOAST_DISMISS_MS,
  TOAST_DEQUEUE_MS,
  TOAST_ENQUEUE_MS,
} from '@nemo/common/src/providers/toast/constants';
import { getTransformStyles } from '@nemo/common/src/providers/toast/GetTransformStyles';
import type {
  AddToastFn,
  ToastContextValue,
  ToastDescriptor,
} from '@nemo/common/src/providers/toast/types';
import { ToastContext } from '@nemo/common/src/providers/toast/useToast';
import { Toast } from '@nvidia/foundations-react-core';
import {
  type FC,
  type PropsWithChildren,
  type SyntheticEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

export const ToastProvider: FC<PropsWithChildren> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastDescriptor[]>([]);
  const timeoutRefs = useRef<Record<string, NodeJS.Timeout>>({});
  const containerRef = useRef<HTMLDivElement>(null);

  // KUI v1 modals use the native <dialog> element via showModal(), which puts
  // them in the browser's top layer. The top layer ignores z-index from the
  // normal stacking context, so toasts rendered as ordinary fixed-position
  // children disappear behind any open modal's backdrop.
  //
  // Fix: render the toast container as a manual Popover and call .showPopover()
  // imperatively at addToast time. Popovers also enter the top layer, and
  // "last opened wins" — re-issuing showPopover() promotes the container above
  // any dialog that opened after the previous call. With no toasts, the
  // popover stays open but renders nothing visible (no children inside).
  const promoteToTopLayer = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    if (el.matches(':popover-open')) el.hidePopover();
    el.showPopover();
  }, []);

  // Cleanup all timeouts when component unmounts to prevent state updates after unmount
  useEffect(() => {
    return () => {
      // Clear all pending timeouts to prevent state updates after unmount
      Object.values(timeoutRefs.current).forEach((timeout) => {
        clearTimeout(timeout);
      });
      timeoutRefs.current = {};
    };
  }, []);

  const stopToastClickPropagation = useCallback((e: SyntheticEvent) => {
    e.stopPropagation();
  }, []);

  const removeToast = useCallback((id: string) => {
    // Clear any existing timeouts for this toast
    if (timeoutRefs.current[id]) {
      clearTimeout(timeoutRefs.current[id]);
      delete timeoutRefs.current[id];
    }
    const dismissKey = `dismiss-${id}`;
    if (timeoutRefs.current[dismissKey]) {
      clearTimeout(timeoutRefs.current[dismissKey]);
      delete timeoutRefs.current[dismissKey];
    }

    setToasts((prevToasts) =>
      prevToasts.map((toast) => (toast.id === id ? { ...toast, isVisible: false } : toast))
    );

    // Use a new timeout for removal
    timeoutRefs.current[id] = setTimeout(() => {
      setToasts((prevToasts) => prevToasts.filter((toast) => toast.id !== id));
      delete timeoutRefs.current[id];
    }, TOAST_DEQUEUE_MS);
  }, []);

  const addToast: AddToastFn = useCallback(
    (message, options) => {
      const { durationMs: rawDurationMs, status } = options;
      const durationMs = rawDurationMs === false ? undefined : rawDurationMs;
      const newToastId = `toast-${crypto.randomUUID()}`;

      promoteToTopLayer();
      setToasts((prevToasts) => [
        ...prevToasts,
        {
          id: newToastId,
          message,
          status,
          isVisible: false,
        },
      ]);

      // Use a single timeout for visibility and auto-dismiss
      timeoutRefs.current[newToastId] = setTimeout(() => {
        setToasts((prevToasts) =>
          prevToasts.map((toast) =>
            toast.id === newToastId ? { ...toast, isVisible: true } : toast
          )
        );

        if (durationMs) {
          timeoutRefs.current[`dismiss-${newToastId}`] = setTimeout(() => {
            removeToast(newToastId);
          }, durationMs);
        }
      }, TOAST_ENQUEUE_MS);

      return newToastId;
    },
    [promoteToTopLayer, removeToast]
  );

  const contextValue: ToastContextValue = useMemo(
    () => ({
      addToast,
      dismissToast: removeToast,
      toast: {
        success: (message, options) =>
          addToast(message, {
            status: 'success',
            durationMs: DEFAULT_TOAST_DISMISS_MS,
            ...options,
          }),
        error: (message, options) =>
          addToast(message, { status: 'error', durationMs: ERROR_TOAST_DISMISS_MS, ...options }),
        info: (message, options) =>
          addToast(message, { status: 'info', durationMs: DEFAULT_TOAST_DISMISS_MS, ...options }),
        warning: (message, options) =>
          addToast(message, {
            status: 'warning',
            durationMs: DEFAULT_TOAST_DISMISS_MS,
            ...options,
          }),
        working: (message, options) => addToast(message, { status: 'working', ...options }),
        workingWithId: (message, options) => addToast(message, { status: 'working', ...options }),
        neutral: (message, options) =>
          addToast(message, { durationMs: DEFAULT_TOAST_DISMISS_MS, ...options }),
        dismissToast: removeToast,
      },
    }),
    [addToast, removeToast]
  );

  return (
    <ToastContext.Provider value={contextValue}>
      {children}
      <div
        ref={containerRef}
        popover="manual"
        // UA stylesheet on `[popover]` sets `display: none` until open and
        // `inset: 0; margin: auto` (centers the element). `left-auto
        // bottom-auto m-0` releases those so `top-...` + `right-4` anchor to
        // the upper-right corner; `[&:popover-open]:flex` restores our layout.
        className="fixed top-[calc(var(--nv-app-bar-height)+1rem)] right-4 left-auto bottom-auto m-0 hidden flex-col items-end gap-4 bg-transparent p-0 z-1100 max-w-md [&:popover-open]:flex"
      >
        {toasts.map((toast) => (
          <Toast
            key={toast.id}
            status={toast.status}
            onClick={stopToastClickPropagation}
            onPointerDown={stopToastClickPropagation}
            onClose={() => removeToast(toast.id)}
            role="alert"
            className={`transition-all duration-300 ${
              !toast.isVisible ? 'opacity-0' : 'opacity-100'
            }`}
            attributes={{
              ToastContent: {
                style: { whiteSpace: 'normal', overflow: 'visible' },
              },
              ToastText: {
                style: {
                  whiteSpace: 'normal',
                  overflow: 'visible',
                  textOverflow: 'clip',
                  wordBreak: 'break-word',
                },
              },
            }}
            /* eslint-disable-next-line no-restricted-syntax */
            style={{
              transform: getTransformStyles({ isVisible: toast.isVisible }),
              height: 'auto',
            }}
          >
            {toast.message}
          </Toast>
        ))}
      </div>
    </ToastContext.Provider>
  );
};
