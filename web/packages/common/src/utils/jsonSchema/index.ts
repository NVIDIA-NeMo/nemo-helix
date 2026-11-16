// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export { canonicalJson } from '@nemo/common/src/utils/jsonSchema/canonical';
export { INFERENCE_MAX_DEPTH, inferJsonSchema } from '@nemo/common/src/utils/jsonSchema/inference';
export type { InferOptions, JsonSchema } from '@nemo/common/src/utils/jsonSchema/inference';
export {
  parseAndValidate,
  validateJsonSchemaDocument,
} from '@nemo/common/src/utils/jsonSchema/validate';
export type {
  ParseAndValidateResult,
  ValidationResult,
} from '@nemo/common/src/utils/jsonSchema/validate';
export { buildDatasetMetadata } from '@nemo/common/src/utils/jsonSchema/dedupe';
export type { PerFileInferred } from '@nemo/common/src/utils/jsonSchema/dedupe';
export { isSchemaAssignableFile } from '@nemo/common/src/utils/jsonSchema/schemaAssignable';
