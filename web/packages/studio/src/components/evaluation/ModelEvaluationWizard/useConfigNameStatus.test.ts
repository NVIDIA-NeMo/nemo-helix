// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema/FilesetPurpose';
import { configNameTakenMessage } from '@studio/components/evaluation/ModelEvaluationWizard/useConfigNameStatus';

describe('configNameTakenMessage', () => {
  it.each([
    [FilesetPurpose.dataset, 'a dataset'],
    [FilesetPurpose.model, 'a model'],
    [FilesetPurpose.generic, 'a fileset'],
    [undefined, 'a fileset'],
  ])('names a %s conflict as %s, never as a configuration', (purpose, owner) => {
    const message = configNameTakenMessage(purpose, 'default');
    expect(message).toBe(
      `This name is already used by ${owner} in the default workspace. Choose a different configuration name.`
    );
    expect(message).not.toMatch(/configuration named/);
  });
});
