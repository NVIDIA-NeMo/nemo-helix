// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  buildMetaSchemaValidatorSources,
  DTS_PATH,
  JS_PATH,
} from '@nemo/common/scripts/metaSchemaValidatorSources';
import { writeFileSync } from 'node:fs';

const { js, dts } = buildMetaSchemaValidatorSources();
writeFileSync(JS_PATH, js);
writeFileSync(DTS_PATH, dts);

// eslint-disable-next-line no-console
console.log(`Meta-schema validator written to ${JS_PATH} and ${DTS_PATH}`);
