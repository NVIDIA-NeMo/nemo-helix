// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import {
  getAuthGetMeQueryKey,
  useAuthUpdateMyProfile,
} from '@nemo/sdk/generated/platform/profiles';
import type { Me, ProfileUpdate } from '@nemo/sdk/generated/platform/schema';
import {
  Avatar,
  Banner,
  Button,
  Combobox,
  Divider,
  Flex,
  FormField,
  PageHeader,
  Panel,
  SegmentedControl,
  Spinner,
  Stack,
  Text,
  TextInput,
  type Theme,
  useTheme,
} from '@nvidia/foundations-react-core';
import { useMe } from '@studio/providers/auth/useMe';
import { studioPreferences } from '@studio/providers/preferences/definitions';
import { usePreference } from '@studio/providers/preferences/usePreference';
import {
  BROWSER_DEFAULT,
  localeOptions,
  timezoneOptions,
} from '@studio/routes/ProfileRoute/options';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, type FormEvent, type ReactNode, useMemo, useState } from 'react';

const MAX_DISPLAY_NAME_LENGTH = 256;

const AUTH_METHOD_LABELS: Record<Me['auth_method'], string> = {
  web_session: 'Browser session',
  oidc_bearer: 'OIDC token',
  access_key: 'Access key',
  workload_token: 'Workload token',
  trusted_header: 'Trusted header',
  disabled: 'Authentication disabled',
};

const THEME_ITEMS = [
  { value: 'light', children: 'Light' },
  { value: 'dark', children: 'Dark' },
  { value: 'system', children: 'System' },
];

interface SectionProps {
  title: string;
  description: string;
  children: ReactNode;
}

const Section: FC<SectionProps> = ({ title, description, children }) => (
  <Flex gap="density-3xl" className="max-md:flex-col">
    <Stack gap="density-xs" className="md:w-[280px] shrink-0">
      <Text kind="body/bold/xl">{title}</Text>
      <Text kind="body/regular/md" className="text-subtle">
        {description}
      </Text>
    </Stack>
    <Stack gap="density-xl" className="flex-1 max-w-[560px]">
      {children}
    </Stack>
  </Flex>
);

const Detail: FC<{ label: string; children: ReactNode }> = ({ label, children }) => (
  <Stack gap="density-xxs">
    <Text kind="label/bold/sm" className="text-subtle">
      {label}
    </Text>
    <Text kind="body/regular/md" className="break-all">
      {children}
    </Text>
  </Stack>
);

const formFromMe = (me: Me) => ({
  display_name: me.profile.display_name_override ?? '',
  timezone: me.profile.timezone ?? BROWSER_DEFAULT,
  locale: me.profile.locale ?? BROWSER_DEFAULT,
});

type ProfileForm = ReturnType<typeof formFromMe>;

/** Only changed fields are sent; empty or browser-default values clear the stored field. */
const toUpdate = (form: ProfileForm, saved: ProfileForm): ProfileUpdate =>
  Object.fromEntries(
    (Object.keys(form) as (keyof ProfileForm)[])
      .filter((key) => form[key] !== saved[key])
      .map((key) => {
        const value = form[key].trim();
        return [key, value && value !== BROWSER_DEFAULT ? value : null];
      })
  );

const ProfileDetails: FC<{ me: Me }> = ({ me }) => {
  const queryClient = useQueryClient();
  const saved = useMemo(() => formFromMe(me), [me]);
  const [form, setForm] = useState(saved);
  const { mutateAsync, isPending, error } = useAuthUpdateMyProfile();
  const update = toUpdate(form, saved);
  const dirty = Object.keys(update).length > 0;
  const readOnly = !me.can_edit_profile;
  const timezones = useMemo(() => timezoneOptions(me.profile.timezone), [me.profile.timezone]);
  const locales = useMemo(() => localeOptions(me.profile.locale), [me.profile.locale]);

  const set = (key: keyof ProfileForm) => (value: string) =>
    setForm((prev) => ({ ...prev, [key]: value }));

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const updated = await mutateAsync({ data: update });
    queryClient.setQueryData(getAuthGetMeQueryKey(), updated);
    setForm(formFromMe(updated));
  };

  return (
    <form onSubmit={onSubmit}>
      <Stack gap="density-xl">
        <FormField
          slotLabel="Display name"
          slotHelp="Shown to other members next to your sign-in identity. Leave empty to use the name from your identity provider."
        >
          <TextInput
            value={form.display_name}
            maxLength={MAX_DISPLAY_NAME_LENGTH}
            placeholder={me.profile.idp_display_name ?? me.email ?? undefined}
            disabled={readOnly}
            onChange={(e) => set('display_name')(e.target.value)}
          />
        </FormField>
        <FormField slotLabel="Time zone" slotHelp="Used when showing dates and times.">
          <Combobox
            items={timezones}
            value={form.timezone}
            onValueChange={set('timezone')}
            disabled={readOnly}
            emptyStateMessage="No matching time zones"
          />
        </FormField>
        <FormField slotLabel="Language and region" slotHelp="Used for number and date formats.">
          <Combobox
            items={locales}
            value={form.locale}
            onValueChange={set('locale')}
            disabled={readOnly}
            emptyStateMessage="No matching languages"
          />
        </FormField>
        {error && (
          <Banner kind="inline" status="error">
            Could not update your profile. Try again.
          </Banner>
        )}
        {!readOnly && (
          <Flex gap="density-md">
            <Button type="submit" color="brand" disabled={!dirty || isPending}>
              {isPending ? <Spinner size="small" aria-label="Saving" /> : 'Save changes'}
            </Button>
            <Button
              type="button"
              color="neutral"
              kind="secondary"
              disabled={!dirty || isPending}
              onClick={() => setForm(saved)}
            >
              Discard
            </Button>
          </Flex>
        )}
      </Stack>
    </form>
  );
};

const Appearance: FC = () => {
  const { setTheme } = useTheme();
  const [theme, setSavedTheme] = usePreference(studioPreferences.theme);
  const [tourSeen, setTourSeen] = usePreference(studioPreferences.tourSeen);

  return (
    <>
      <FormField slotLabel="Theme">
        <SegmentedControl
          items={THEME_ITEMS}
          value={theme}
          onValueChange={(next) => {
            setTheme(next as Theme);
            setSavedTheme(next as Theme);
          }}
        />
      </FormField>
      <Flex justify="between" align="center" gap="density-xl">
        <Stack gap="density-xxs">
          <Text kind="label/bold/md">Welcome tour</Text>
          <Text kind="body/regular/sm" className="text-subtle">
            {tourSeen
              ? 'Show the Studio tour again the next time you open a workspace.'
              : 'The tour will show the next time you open a workspace.'}
          </Text>
        </Stack>
        <Button
          color="neutral"
          kind="secondary"
          disabled={!tourSeen}
          onClick={() => setTourSeen(false)}
        >
          Restart tour
        </Button>
      </Flex>
    </>
  );
};

export const ProfileRoute: FC = () => {
  const { data: me, isLoading, isError } = useMe();

  return (
    <AccessibleTitle title="Profile">
      <Stack gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading="Profile"
          slotDescription="Manage how you appear to others and personalize Studio."
        />
        {isLoading && <Spinner aria-label="Loading profile" />}
        {isError && (
          <Banner kind="inline" status="error">
            Could not load your profile.
          </Banner>
        )}
        {me && (
          <Panel elevation="high">
            <Stack gap="10">
              <Section
                title="Account"
                description="Identity details from your sign-in provider. These cannot be changed here."
              >
                <Flex gap="density-lg" align="center">
                  <Avatar
                    size="large"
                    fallback={me.profile.display_name.charAt(0) || 'N'}
                    src={me.profile.avatar_url ?? undefined}
                    alt=""
                    referrerPolicy="no-referrer"
                  />
                  <Stack gap="density-xxs">
                    <Text kind="body/bold/lg">{me.profile.display_name}</Text>
                    {me.email && (
                      <Text kind="body/regular/md" className="text-subtle">
                        {me.email}
                      </Text>
                    )}
                  </Stack>
                </Flex>
                <Detail label="Principal">{me.principal_id}</Detail>
                <Detail label="Signed in with">{AUTH_METHOD_LABELS[me.auth_method]}</Detail>
                {me.groups?.length ? <Detail label="Groups">{me.groups.join(', ')}</Detail> : null}
              </Section>
              <Divider />
              <Section
                title="Profile"
                description={
                  me.can_edit_profile
                    ? 'Your name and regional settings, shared across Studio, the CLI, and the SDK.'
                    : 'This session cannot change your profile.'
                }
              >
                <ProfileDetails me={me} />
              </Section>
              <Divider />
              <Section
                title="Appearance"
                description="Studio preferences. Changes apply immediately."
              >
                <Appearance />
              </Section>
            </Stack>
          </Panel>
        )}
      </Stack>
    </AccessibleTitle>
  );
};
