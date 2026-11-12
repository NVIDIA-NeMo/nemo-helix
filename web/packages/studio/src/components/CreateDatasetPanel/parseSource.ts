// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type SourceTab = 'huggingface' | 'ngc' | 'github' | 's3';

export interface HuggingFaceFields {
  repo_id: string;
  repo_type: 'dataset' | 'model' | 'space';
  revision: string;
}
export interface NgcFields {
  org: string;
  team: string;
  target: string;
  target_type: 'resource' | 'model';
  version?: string;
}
export interface GithubFields {
  owner: string;
  repo: string;
  revision: string;
  path?: string;
}
export interface S3Fields {
  bucket: string;
  prefix?: string;
}
export type SourceFields = HuggingFaceFields | NgcFields | GithubFields | S3Fields;

export type ParseResult =
  | { status: 'empty' }
  | { status: 'invalid' }
  | { status: 'wrong-tab'; actual: SourceTab }
  | { status: 'incomplete' }
  | { status: 'ok'; fields: SourceFields; nameSeed: string };

export function parseSource(_tab: SourceTab, _raw: string): ParseResult {
  throw new Error('not implemented');
}
