// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  Me,
  PreferenceDocument,
  ProfileResolveRequest,
  ProfileResolveResponse,
} from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { http, HttpResponse } from 'msw';

export const mockMe: Me = {
  account_id: 'account-test',
  principal_id: 'test-user',
  email: 'test.user@example.com',
  groups: [],
  caller_kind: 'principal',
  auth_method: 'oidc_bearer',
  profile: { display_name: 'test.user@example.com' },
  can_edit_profile: true,
  is_platform_admin: false,
};

const emptyDocument = (namespace: string, workspace?: string): PreferenceDocument => ({
  namespace,
  workspace,
  value: {},
  defaults: {},
  effective: {},
  db_version: 0,
});

export const profilesHandlers = [
  http.get(`${PLATFORM_BASE_URL}/apis/auth/v2/me`, () => HttpResponse.json(mockMe)),
  http.get(`${PLATFORM_BASE_URL}/apis/auth/v2/me/preferences/:namespace`, ({ params, request }) =>
    HttpResponse.json(
      emptyDocument(
        String(params.namespace),
        new URL(request.url).searchParams.get('workspace') ?? undefined
      )
    )
  ),
  // Unresolved by default, so tables render raw principals unless a test opts in.
  http.post(`${PLATFORM_BASE_URL}/apis/auth/v2/profiles/resolve`, async ({ request }) => {
    const { principals } = (await request.json()) as ProfileResolveRequest;
    const response: ProfileResolveResponse = {
      data: principals.map((principal) => ({
        principal,
        display_name_is_override: false,
        ambiguous: false,
      })),
    };
    return HttpResponse.json(response);
  }),
];
