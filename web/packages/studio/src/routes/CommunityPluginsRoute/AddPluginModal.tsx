// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  useInstallPluginApisPluginsInstalledPost,
  usePreviewPluginApisPluginsPreviewPost,
} from '@nemo/sdk/generated/platform/external-plugins';
import type { PreviewResponse } from '@nemo/sdk/generated/platform/schema';
import { Button, Checkbox, FormField, TextInput } from '@nvidia/foundations-react-core';
import { PluginPreview } from '@studio/routes/CommunityPluginsRoute/PluginPreview';
import { invalidatePluginCaches } from '@studio/routes/CommunityPluginsRoute/pluginQueries';
import { useQueryClient } from '@tanstack/react-query';
import { AxiosError } from 'axios';
import { type FC, type FormEvent, useState } from 'react';

interface AddPluginModalProps {
  /** Prefills the URL, for catalog suggestions. */
  initialUrl: string;
  onClose: () => void;
}

/** What the administrator previewed. Install sends exactly this, not whatever the field says now. */
interface Previewed extends PreviewResponse {
  url: string;
}

/**
 * Two steps. Preview fetches and validates the manifest without installing anything.
 * Install sends the previewed URL with its SHA-256 and an explicit consent flag, and the
 * backend refuses (409) if the manifest changed in between, so what was reviewed is
 * what gets registered.
 */
export const AddPluginModal: FC<AddPluginModalProps> = ({ initialUrl, onClose }) => {
  const toast = useToast();
  const queryClient = useQueryClient();
  const [url, setUrl] = useState(initialUrl);
  const [previewed, setPreviewed] = useState<Previewed>();
  const [consent, setConsent] = useState(false);
  const [error, setError] = useState<string>();

  const previewPlugin = usePreviewPluginApisPluginsPreviewPost();
  const installPlugin = useInstallPluginApisPluginsInstalledPost();

  const backToUrl = () => {
    setPreviewed(undefined);
    setConsent(false);
    setError(undefined);
  };

  const handlePreview = async () => {
    setError(undefined);
    const target = url.trim();
    try {
      const preview = await previewPlugin.mutateAsync({ data: { url: target } });
      setPreviewed({ ...preview, url: target });
    } catch (err) {
      setError(getErrorMessage(err as Error));
    }
  };

  const handleInstall = async (current: Previewed) => {
    setError(undefined);
    try {
      await installPlugin.mutateAsync({
        data: { url: current.url, manifest_sha256: current.manifest_sha256, consent: true },
      });
      await invalidatePluginCaches(queryClient);
      toast.success(`${current.manifest.plugin.title} registered`);
      onClose();
    } catch (err) {
      const message = getErrorMessage(err as Error);
      if (err instanceof AxiosError && err.response?.status === 409) {
        // The manifest changed, or the name is taken: the review no longer applies.
        backToUrl();
      }
      setError(message);
    }
  };

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void (previewed ? handleInstall(previewed) : handlePreview());
  };

  if (!previewed) {
    return (
      <FormModal
        open
        title="Add Plugin"
        instruction="Enter the URL of the plugin's manifest, in YAML or JSON. Nothing is installed until you review it."
        submitButtonText="Preview"
        errorText={error}
        loading={previewPlugin.isPending}
        submitDisabled={url.trim() === ''}
        onSubmit={onSubmit}
        onClose={onClose}
      >
        <FormField slotLabel="Manifest URL">
          <TextInput
            value={url}
            onValueChange={setUrl}
            placeholder="https://example.com/plugin.yaml"
            aria-label="Manifest URL"
          />
        </FormField>
      </FormModal>
    );
  }

  const hasUi = Boolean(previewed.manifest.plugin.bundle);
  return (
    <FormModal
      open
      title={`Review: ${previewed.manifest.plugin.title}`}
      submitButtonText="Register"
      errorText={error}
      loading={installPlugin.isPending}
      submitDisabled={!consent}
      onSubmit={onSubmit}
      onClose={onClose}
      className="w-[90vw] max-w-[880px]"
      slotFooterLeft={
        <Button kind="tertiary" type="button" onClick={backToUrl}>
          Back
        </Button>
      }
    >
      <PluginPreview preview={previewed} />
      <Checkbox
        checked={consent}
        onChange={(event) => setConsent(event.target.checked)}
        slotLabel={
          hasUi
            ? 'I have reviewed this plugin, and I approve exposing its API routes and running its UI code in Studio.'
            : 'I have reviewed this plugin, and I approve exposing its API routes.'
        }
      />
    </FormModal>
  );
};
