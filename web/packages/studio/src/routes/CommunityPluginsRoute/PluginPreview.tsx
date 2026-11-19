// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PreviewResponse } from '@nemo/sdk/generated/platform/schema';
import { Banner, Stack, Text } from '@nvidia/foundations-react-core';
import type { FC, ReactNode } from 'react';

const Section: FC<{ title: string; children: ReactNode }> = ({ title, children }) => (
  <Stack gap="density-xs">
    <Text kind="label/bold/md">{title}</Text>
    {children}
  </Stack>
);

const Mono: FC<{ children: ReactNode }> = ({ children }) => (
  <code className="break-all text-xs">{children}</code>
);

/**
 * Everything the administrator consents to: the API routes the platform will expose and
 * where they go, the secret it will send, and the UI code Studio will run. It renders the
 * validated manifest the backend returned, never the raw text the publisher served.
 */
export const PluginPreview: FC<{ preview: PreviewResponse }> = ({ preview }) => {
  const { plugin } = preview.manifest;
  const { bundle } = plugin;

  return (
    <Stack gap="density-lg">
      {preview.warning && (
        <Banner status="warning" kind="inline" data-testid="plugin-trust-warning">
          {preview.warning}
        </Banner>
      )}

      <Section title="Manifest">
        <Text kind="body/regular/sm">
          {plugin.title} (<Mono>{plugin.name}</Mono>)
        </Text>
        <Text kind="body/regular/xs" className="text-subtle">
          SHA-256 <Mono>{preview.manifest_sha256}</Mono>
        </Text>
      </Section>

      <Section title={`API endpoints (${plugin.endpoints.length})`}>
        <table className="w-full text-left text-sm" data-testid="plugin-endpoints">
          <thead className="text-subtle">
            <tr>
              <th className="pr-3 font-medium">Method</th>
              <th className="pr-3 font-medium">Studio calls</th>
              <th className="pr-3 font-medium">Access</th>
              <th className="font-medium">Forwarded to</th>
            </tr>
          </thead>
          <tbody>
            {plugin.endpoints.map((endpoint) => (
              <tr key={`${endpoint.method} ${endpoint.path}`} className="align-top">
                <td className="pr-3">{endpoint.method}</td>
                <td className="pr-3">
                  <Mono>{`/apis/plugins/${plugin.name}${endpoint.path}`}</Mono>
                </td>
                <td className="pr-3">{endpoint.access}</td>
                <td>
                  <Mono>{endpoint.upstream_url}</Mono>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>

      {plugin.credential && (
        <Section title="Credential">
          <Text kind="body/regular/sm">
            Sends the workspace secret <Mono>{plugin.credential.secret_name}</Mono> upstream as a{' '}
            {plugin.credential.scheme ?? 'bearer'} token. Your own Studio credentials are never
            forwarded.
          </Text>
        </Section>
      )}

      {bundle ? (
        <Section title="Studio UI">
          <Text kind="body/regular/sm">
            Studio will download this bundle, pin its hash, and run it inside Studio itself, with
            the same access as Studio. It is not sandboxed.
          </Text>
          <Text kind="body/regular/xs" className="text-subtle">
            Bundle <Mono>{bundle.url}</Mono>
          </Text>
          <Text kind="body/regular/xs" className="text-subtle">
            SHA-256 <Mono>{bundle.sha256}</Mono>
          </Text>
          <Text kind="body/regular/xs" className="text-subtle">
            Built for plugin contract {bundle.sidecar.contractVersion}
          </Text>
          {(plugin.navigation ?? []).map((item) => (
            <Text key={item.path} kind="body/regular/xs" className="text-subtle">
              Side nav: {item.section ? `${item.section} › ` : ''}
              {item.label} (<Mono>{item.path}</Mono>)
            </Text>
          ))}
        </Section>
      ) : (
        <Section title="Studio UI">
          <Text kind="body/regular/sm">None. This plugin only adds API routes.</Text>
        </Section>
      )}
    </Stack>
  );
};
