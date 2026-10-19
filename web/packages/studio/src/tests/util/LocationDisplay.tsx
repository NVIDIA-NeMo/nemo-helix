// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { FC } from 'react';
import { useLocation } from 'react-router';

/**
 * Useful for asserting a redirect happened in a test, for example:
 *
 * ```
 * render(<CustomizationJobListRoute />, {
 *   initialPath: ROUTES.model.index,
 *   otherRoutes: [
 *     {
 *       path: ROUTES.workspace.newCustomizationJob,
 *       element: <LocationDisplay />,
 *     },
 *   ],
 * });
 *
 *
 * // Assert user was redirected to the new Model's playground route
 * const location = (await screen.findByTestId(LOCATION_DISPLAY_TEST_ID)).textContent;
 * expect(location).toEqual(getNewCustomizationJobRoute(workspace, model1));
 * ```
 *
 * See: https://testing-library.com/docs/example-react-router/
 */
export const LocationDisplay: FC = () => {
  const location = useLocation();

  return (
    <div data-testid={LOCATION_DISPLAY_TEST_ID}>
      {location.pathname}
      {location.search}
    </div>
  );
};
