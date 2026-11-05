// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetOutput, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { Anchor } from '@nvidia/foundations-react-core';
import { TagList } from '@studio/routes/FilesetDetailRoute/FilesetMetadataPanel/TagList';
import { getModelSource } from '@studio/routes/FilesetDetailRoute/utils';
import { getWorkspaceDeploymentsRoute } from '@studio/routes/utils';
import { getFormattedTrainingType } from '@studio/util/customizations';
import { formatStorageBackendLabel } from '@studio/util/storageBackend';
import type { ReactNode } from 'react';

interface MetadataRow {
  label: string;
  value: ReactNode;
}

interface MetadataSection {
  value: string;
  title: string;
  rows: MetadataRow[];
}

export const getMetadataSections = (
  fileset: FilesetOutput,
  readmeMetadata?: Record<string, unknown>,
  modelEntities?: ModelEntity[]
): MetadataSection[] => {
  const sections: MetadataSection[] = [
    { value: 'source', title: 'Source', rows: getSourceRows(fileset) },
  ];

  const detailsRows = getModelCardRows(readmeMetadata);
  if (detailsRows.length > 0) {
    sections.push({ value: 'details', title: 'Details', rows: detailsRows });
  }

  (modelEntities ?? []).forEach((entity, index) => {
    sections.push(getModelEntitySection(entity, index));
  });

  return sections;
};

const getSourceRows = (fileset: FilesetOutput): MetadataRow[] => {
  const { storage } = fileset;
  const source = getModelSource(fileset);

  const rows: MetadataRow[] = [
    { label: 'Storage', value: formatStorageBackendLabel(storage.type) ?? 'Unknown' },
  ];

  if (source) {
    rows.push({
      label: 'Source',
      value:
        storage.type === 'huggingface' ? (
          <Anchor
            href={`https://huggingface.co/${source.path}`}
            target="_blank"
            rel="noopener noreferrer"
          >
            {source.path}
          </Anchor>
        ) : (
          source.path
        ),
    });
  }

  if (storage.type === 'huggingface' && storage.revision) {
    rows.push({ label: 'Revision', value: storage.revision });
  } else if (storage.type === 'ngc' && 'target' in storage) {
    if (storage.team) rows.push({ label: 'Team', value: storage.team });
    if (storage.version) rows.push({ label: 'Version', value: storage.version });
  } else if (storage.type === 's3' && 'bucket' in storage) {
    if (storage.region) rows.push({ label: 'Region', value: storage.region });
    if (storage.prefix) rows.push({ label: 'Prefix', value: storage.prefix });
  }

  return rows;
};

const getModelCardRows = (metadata: Record<string, unknown> | undefined): MetadataRow[] => {
  if (!metadata) return [];

  const rows: MetadataRow[] = [];

  const license = readString(metadata.license);
  const licenseLink = readString(metadata.license_link);
  if (license) {
    // `license_link` comes from user-supplied README frontmatter; gate it
    // through `isSafeHttpUrl` to block `javascript:` / `data:` URLs.
    const safeLicenseLink = licenseLink && isSafeHttpUrl(licenseLink) ? licenseLink : undefined;
    rows.push({
      label: 'License',
      value: safeLicenseLink ? (
        <Anchor href={safeLicenseLink} target="_blank" rel="noopener noreferrer">
          {license}
        </Anchor>
      ) : (
        license
      ),
    });
  }

  const library = readString(metadata.library_name);
  if (library) rows.push({ label: 'Library', value: library });

  const pipeline = readString(metadata.pipeline_tag);
  if (pipeline) rows.push({ label: 'Task', value: pipeline });

  const baseModel = readString(metadata.base_model);
  if (baseModel) {
    rows.push({
      label: 'Based on',
      value: looksLikeHuggingFaceRepo(baseModel) ? (
        <Anchor
          href={`https://huggingface.co/${baseModel}`}
          target="_blank"
          rel="noopener noreferrer"
        >
          {baseModel}
        </Anchor>
      ) : (
        baseModel
      ),
    });
  }

  const languages = readStringList(metadata.language ?? metadata.languages);
  if (languages.length > 0) {
    rows.push({
      label: languages.length === 1 ? 'Language' : 'Languages',
      value: <TagList items={languages} />,
    });
  }

  const tags = readStringList(metadata.tags);
  if (tags.length > 0) {
    rows.push({ label: 'Tags', value: <TagList items={tags} /> });
  }

  return rows;
};

const readString = (value: unknown): string | undefined => {
  if (typeof value === 'string' && value.trim().length > 0) return value.trim();
  return undefined;
};

const readStringList = (value: unknown): string[] => {
  if (typeof value === 'string') {
    const single = readString(value);
    return single ? [single] : [];
  }
  if (Array.isArray(value)) {
    return value.filter(
      (item): item is string => typeof item === 'string' && item.trim().length > 0
    );
  }
  return [];
};

const looksLikeHuggingFaceRepo = (value: string): boolean =>
  /^[\w.-]+\/[\w.-]+$/.test(value) && !value.startsWith('http');

const isSafeHttpUrl = (value: string): boolean => {
  try {
    const { protocol } = new URL(value);
    return protocol === 'http:' || protocol === 'https:';
  } catch {
    return false;
  }
};

const getModelEntitySection = (entity: ModelEntity, index: number): MetadataSection => {
  const rows: MetadataRow[] = [];

  if (entity.description) {
    rows.push({ label: 'Description', value: entity.description });
  }

  if (entity.base_model) {
    rows.push({ label: 'Base model', value: entity.base_model });
  }

  if (entity.finetuning_type) {
    rows.push({ label: 'Fine-tuning', value: getFormattedTrainingType(entity.finetuning_type) });
  }

  const isDeployed = (entity.model_providers?.length ?? 0) > 0;
  rows.push({
    label: 'Deployment',
    value: isDeployed ? (
      <Anchor href={getWorkspaceDeploymentsRoute(entity.workspace)} target="_self">
        Deployed
      </Anchor>
    ) : (
      'Not deployed'
    ),
  });

  return {
    value: `model-entity-${index}`,
    title: `Model Entity ${index + 1}: ${entity.name}`,
    rows,
  };
};
