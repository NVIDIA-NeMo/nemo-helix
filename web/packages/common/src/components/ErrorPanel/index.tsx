// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage, isNotFoundError } from '@nemo/common/src/api/common/utils';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { logger } from '@nemo/common/src/utils/logger';
import { Button, Flex, PageHeader, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import { useQueryClient } from '@tanstack/react-query';
import { AxiosError } from 'axios';
import { ComponentProps, FC, ReactNode, useEffect } from 'react';
import { isRouteErrorResponse, useLocation, useNavigate, useRouteError } from 'react-router';

const DEFAULT_ERROR_MESSAGE =
  'An unexpected error occurred.\nPlease contact your Administrator for support.';

export interface ErrorPanelProps {
  /** The title displayed in the header */
  title?: ReactNode;
  /** The error message to display. If you need greater control over the error message, use attributes.ErrorMessage to pass props to the ErrorMessage component directly. */
  errorMessage?: string;
  /** Props to pass to the ErrorMessage component */
  attributes?: {
    ErrorMessage?: ComponentProps<typeof ErrorMessage>;
  };
  /** The underlying error, when known (e.g. resolved by `RouteErrorPanel` from `useRouteError`). */
  error?: unknown;
  /** Shows a Retry action. Not shown for not-found or forbidden errors, which retrying won't fix. */
  onRetry?: () => void;
}

const isForbidden = (error: unknown) =>
  (isRouteErrorResponse(error) && error.status === 403) ||
  (error instanceof AxiosError && error.response?.status === 403);

/**
 * Extracts a user-friendly error message from various error types.
 */
const getRouterErrorMessage = (error: unknown): string | undefined => {
  if (isRouteErrorResponse(error)) {
    return error.statusText || `Error ${error.status}`;
  } else if (error instanceof Error) {
    return getErrorMessage(error);
  }

  return undefined;
};

const getErrorContent = (error: unknown, errorMessage?: string) => {
  if (isNotFoundError(error)) {
    return {
      header: 'Not Found',
      message:
        errorMessage ??
        "This doesn't exist or may have been deleted. Check the link and try again.",
      retryable: false,
    };
  }
  if (isForbidden(error)) {
    return {
      header: 'No Access',
      message:
        errorMessage ?? "You don't have permission to view this. Ask a workspace admin for access.",
      retryable: false,
    };
  }
  return {
    header: "Couldn't Load Data",
    message: errorMessage ?? getRouterErrorMessage(error) ?? DEFAULT_ERROR_MESSAGE,
    retryable: true,
  };
};

/**
 * Generic error panel. Renders from the `error`/`errorMessage` props it is given —
 * it never reads a data-router error itself, so it is safe to render inline (e.g. a DataView's
 * `renderErrorState`) as well as anywhere else outside a route `errorElement`.
 *
 * Not-found (404) and forbidden (403) errors get their own heading and copy.
 *
 * Use {@link RouteErrorPanel} instead when rendering as a route's `errorElement`.
 *
 * @example
 * ```tsx
 * renderErrorState: () => (
 *   <ErrorPanel error={error} onRetry={refetch} />
 * )
 * ```
 */
export const ErrorPanel: FC<ErrorPanelProps> = ({
  title,
  errorMessage,
  attributes,
  error,
  onRetry,
}) => {
  const navigate = useNavigate();
  const { header, message, retryable } = getErrorContent(error, errorMessage);

  return (
    <Stack className="h-full" padding="4" gap="3">
      <PageHeader slotHeading={title} />
      <Panel className="flex-1 justify-center" elevation="high">
        <Flex
          align="center"
          justify="center"
          className="h-full w-full p-8"
          data-testid="error-panel"
        >
          <ErrorMessage
            header={<Text kind="title/md">{header}</Text>}
            message={
              <Text kind="body/regular/md" className="whitespace-pre-wrap">
                {message}
              </Text>
            }
            slotFooter={
              // Refreshing can't fix a not-found or forbidden error, so those only offer Go Back.
              retryable && !onRetry ? undefined : (
                <Flex gap="density-sm">
                  {retryable && (
                    <Button kind="secondary" onClick={onRetry}>
                      Retry
                    </Button>
                  )}
                  <Button kind="secondary" onClick={() => navigate(-1)}>
                    Go Back
                  </Button>
                </Flex>
              )
            }
            {...attributes?.ErrorMessage}
          />
        </Flex>
      </Panel>
    </Stack>
  );
};

/**
 * Route-only wrapper around {@link ErrorPanel}. Resolves the current route error via
 * `useRouteError()` — which is only valid inside a data router's `errorElement` tree — and hands
 * it down to the hook-free `ErrorPanel`.
 *
 * Every route has one of these as its `errorElement`, so a page that loads its primary resource
 * with a generated `use*Suspense` hook gets its loading (from `SuspenseOutlet`), not-found, and
 * error states for free. Retry resets the failed queries and re-renders the route.
 *
 * @example
 * ```tsx
 * // In your route configuration:
 * {
 *   path: ROUTES.workspace.evaluation,
 *   element: <EvaluationLayout />,
 *   errorElement: <RouteErrorPanel title="Evaluator" />,
 * }
 * ```
 */
export const RouteErrorPanel: FC<Omit<ErrorPanelProps, 'error' | 'onRetry'>> = (props) => {
  const error = useRouteError();
  // Inline panels show query errors that are already handled; a route-level error is worth reporting.
  useEffect(() => {
    logger.error(JSON.stringify(props.errorMessage, null, 2), error);
  }, [error, props.errorMessage]);
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const { pathname, search, hash, state } = useLocation();

  const retry = () => {
    void queryClient.resetQueries({ predicate: (query) => query.state.status === 'error' });
    // A new location resets the route's error boundary so the page renders again.
    navigate({ pathname, search, hash }, { replace: true, state });
  };

  return <ErrorPanel {...props} error={error} onRetry={retry} />;
};
