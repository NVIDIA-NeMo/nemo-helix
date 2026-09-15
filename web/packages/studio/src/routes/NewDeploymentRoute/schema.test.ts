// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Engine } from '@nemo/sdk/generated/platform/schema';
import {
  additionalEnvsFormToApi,
  configNameFromWizardBaseName,
  createDeploymentWizardSchema,
  defaultWizardValues,
  deploymentNameFromWizardBaseName,
  engineRequiresImage,
  sourceSupportsEngineChoice,
  WORKSPACE_PICKER_FILESET,
  WORKSPACE_PICKER_MODEL,
  SOURCE_HF,
  SOURCE_WORKSPACE,
  SOURCE_NGC,
  WIZARD_CONFIG_NAME_SUFFIX,
  WIZARD_DEPLOYMENT_NAME_SUFFIX,
} from '@studio/routes/NewDeploymentRoute/schema';

describe('deploymentNameFromWizardBaseName', () => {
  it('appends deployment suffix', () => {
    expect(deploymentNameFromWizardBaseName('my-app')).toBe(
      `my-app${WIZARD_DEPLOYMENT_NAME_SUFFIX}`
    );
  });
});

describe('configNameFromWizardBaseName', () => {
  it('appends config suffix', () => {
    expect(configNameFromWizardBaseName('my-app')).toBe(`my-app${WIZARD_CONFIG_NAME_SUFFIX}`);
  });
});

describe('additionalEnvsFormToApi', () => {
  it('returns undefined for empty rows', () => {
    expect(additionalEnvsFormToApi([])).toBeUndefined();
  });

  it('returns undefined when all keys are blank', () => {
    expect(additionalEnvsFormToApi([{ key: '', value: 'val' }])).toBeUndefined();
  });

  it('builds record from valid rows', () => {
    const rows = [
      { key: 'FOO', value: 'bar' },
      { key: 'BAZ', value: 'qux' },
    ];
    expect(additionalEnvsFormToApi(rows)).toEqual({ FOO: 'bar', BAZ: 'qux' });
  });

  it('trims keys and values', () => {
    const rows = [{ key: ' KEY ', value: ' val ' }];
    expect(additionalEnvsFormToApi(rows)).toEqual({ KEY: 'val' });
  });

  it('uses empty string for undefined value', () => {
    const rows = [{ key: 'KEY' }];
    expect(additionalEnvsFormToApi(rows)).toEqual({ KEY: '' });
  });
});

describe('defaultWizardValues', () => {
  it('returns HuggingFace source with defaults', () => {
    const vals = defaultWizardValues();
    expect(vals.source).toBe(SOURCE_HF);
    expect(vals.gpu).toBe(1);
    expect(vals.loraEnabled).toBe(true);
    expect(typeof vals.name).toBe('string');
    expect(vals.name.length).toBeGreaterThan(0);
  });

  it('defaults the engine to vLLM, which needs no image', () => {
    expect(defaultWizardValues().engine).toBe(Engine.vllm);
    expect(engineRequiresImage(Engine.vllm)).toBe(false);
  });

  it('defaults to a source that actually reads the default engine', () => {
    // The NGC source overrides `engine` to `nim` when building its request, so
    // an NGC default made `engine: vllm` unreachable without switching tabs.
    expect(sourceSupportsEngineChoice(defaultWizardValues().source)).toBe(true);
  });
});

describe('engine image requirements', () => {
  const workspaceValues = (overrides: Record<string, unknown>) => ({
    ...defaultWizardValues(),
    source: SOURCE_WORKSPACE,
    name: 'my-deploy',
    workspacePickerType: WORKSPACE_PICKER_MODEL,
    modelRef: 'ws/model',
    ...overrides,
  });

  it.each([Engine.nim, Engine.generic])('requires an image for the %s engine', (engine) => {
    expect(engineRequiresImage(engine)).toBe(true);

    const result = createDeploymentWizardSchema.safeParse(
      workspaceValues({ engine, imageName: '', imageTag: '' })
    );

    expect(result.success).toBe(false);
    const paths = result.success ? [] : result.error.issues.map((i) => i.path.join('.'));
    expect(paths).toContain('imageName');
    expect(paths).toContain('imageTag');
  });

  it('accepts a blank image for the vLLM engine', () => {
    const result = createDeploymentWizardSchema.safeParse(
      workspaceValues({ engine: Engine.vllm, imageName: '', imageTag: '' })
    );

    expect(result.success).toBe(true);
  });

  it('accepts NIM once an image is supplied', () => {
    const result = createDeploymentWizardSchema.safeParse(
      workspaceValues({
        engine: Engine.nim,
        imageName: 'nvcr.io/nim/meta/llama-3.1-8b-instruct',
        imageTag: '1.8.5',
      })
    );

    expect(result.success).toBe(true);
  });

  it('applies the rule to the HuggingFace source too', () => {
    const result = createDeploymentWizardSchema.safeParse({
      ...defaultWizardValues(),
      source: SOURCE_HF,
      name: 'my-deploy',
      repoId: 'Qwen/Qwen3-0.6B',
      engine: Engine.nim,
      imageName: '',
      imageTag: '',
    });

    expect(result.success).toBe(false);
  });
});

describe('createDeploymentWizardSchema', () => {
  const validNgc = {
    ...defaultWizardValues(),
    source: SOURCE_NGC,
    name: 'test-deploy',
    imageName: 'nvcr.io/nim/test',
    imageTag: 'v1.0',
    gpu: 1,
  };

  const validHf = {
    ...defaultWizardValues(),
    source: SOURCE_HF,
    name: 'test-deploy',
    repoId: 'Qwen/Qwen2.5-1.5B-Instruct',
    gpu: 1,
  };

  it('validates a valid NGC deployment', () => {
    const result = createDeploymentWizardSchema.safeParse(validNgc);
    expect(result.success).toBe(true);
  });

  it('validates a valid HuggingFace deployment', () => {
    const result = createDeploymentWizardSchema.safeParse(validHf);
    expect(result.success).toBe(true);
  });

  it('rejects empty name', () => {
    const result = createDeploymentWizardSchema.safeParse({ ...validNgc, name: '' });
    expect(result.success).toBe(false);
  });

  it('rejects NGC without imageName', () => {
    const result = createDeploymentWizardSchema.safeParse({ ...validNgc, imageName: '' });
    expect(result.success).toBe(false);
  });

  it('rejects NGC without imageTag', () => {
    const result = createDeploymentWizardSchema.safeParse({ ...validNgc, imageTag: '' });
    expect(result.success).toBe(false);
  });

  it('rejects HF without repoId', () => {
    const result = createDeploymentWizardSchema.safeParse({ ...validHf, repoId: '' });
    expect(result.success).toBe(false);
  });

  it('rejects gpu < 1', () => {
    const result = createDeploymentWizardSchema.safeParse({ ...validNgc, gpu: 0 });
    expect(result.success).toBe(false);
  });

  it('detects duplicate additionalEnvs keys', () => {
    const result = createDeploymentWizardSchema.safeParse({
      ...validNgc,
      additionalEnvs: [
        { key: 'FOO', value: '1' },
        { key: 'FOO', value: '2' },
      ],
    });
    expect(result.success).toBe(false);
  });

  describe('workspace source', () => {
    const validWorkspaceModel = {
      ...defaultWizardValues(),
      source: SOURCE_WORKSPACE,
      name: 'test-deploy',
      workspacePickerType: WORKSPACE_PICKER_MODEL,
      modelRef: 'default/my-model',
      gpu: 1,
    };

    const validWorkspaceFileset = {
      ...defaultWizardValues(),
      source: SOURCE_WORKSPACE,
      name: 'test-deploy',
      workspacePickerType: WORKSPACE_PICKER_FILESET,
      fileset: 'default/my-fileset',
      gpu: 1,
    };

    it('validates workspace deployment with model ref', () => {
      const result = createDeploymentWizardSchema.safeParse(validWorkspaceModel);
      expect(result.success).toBe(true);
    });

    it('validates workspace deployment with fileset ref', () => {
      const result = createDeploymentWizardSchema.safeParse(validWorkspaceFileset);
      expect(result.success).toBe(true);
    });

    it('rejects workspace model picker without modelRef', () => {
      const result = createDeploymentWizardSchema.safeParse({
        ...validWorkspaceModel,
        modelRef: '',
      });
      expect(result.success).toBe(false);
    });

    it('rejects workspace fileset picker without fileset', () => {
      const result = createDeploymentWizardSchema.safeParse({
        ...validWorkspaceFileset,
        fileset: '',
      });
      expect(result.success).toBe(false);
    });

    it('rejects modelRef missing the namespace segment', () => {
      const result = createDeploymentWizardSchema.safeParse({
        ...validWorkspaceModel,
        modelRef: 'my-model',
      });
      expect(result.success).toBe(false);
    });

    it('rejects fileset with illegal characters', () => {
      const result = createDeploymentWizardSchema.safeParse({
        ...validWorkspaceFileset,
        fileset: 'default/my fileset',
      });
      expect(result.success).toBe(false);
    });

    it('accepts a deep-linked modelRef of the form `<ws>/<name>`', () => {
      const result = createDeploymentWizardSchema.safeParse({
        ...validWorkspaceModel,
        modelRef: 'steramae/astd-110-test-model',
      });
      expect(result.success).toBe(true);
    });

    it('ignores the inactive picker field', () => {
      // When picker is "model", an empty/missing fileset is fine, and vice versa.
      const result = createDeploymentWizardSchema.safeParse({
        ...validWorkspaceModel,
        fileset: '',
      });
      expect(result.success).toBe(true);
    });
  });
});
