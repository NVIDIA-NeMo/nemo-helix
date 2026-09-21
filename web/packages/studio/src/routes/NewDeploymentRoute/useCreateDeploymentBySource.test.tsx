// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { filesCreateFileset } from '@nemo/sdk/generated/platform/files';
import {
  modelsCreateDeploymentConfig,
  modelsGetLatestDeploymentConfig,
} from '@nemo/sdk/generated/platform/model-deployment-configs';
import { modelsCreateDeployment } from '@nemo/sdk/generated/platform/model-deployments';
import { modelsCreateModel, modelsGetModel } from '@nemo/sdk/generated/platform/models';
import { Engine } from '@nemo/sdk/generated/platform/schema';
import {
  defaultWizardValues,
  WORKSPACE_PICKER_FILESET,
  WORKSPACE_PICKER_MODEL,
  SOURCE_HF,
  SOURCE_WORKSPACE,
  type WizardFormValues,
} from '@studio/routes/NewDeploymentRoute/schema';
import {
  createUnboundDeploymentConfig,
  ensureUnboundDeploymentConfig,
  ensureWorkspaceDeploymentConfig,
  useCreateDeploymentBySource,
} from '@studio/routes/NewDeploymentRoute/useCreateDeploymentBySource';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook } from '@testing-library/react';
import { AxiosError, AxiosHeaders } from 'axios';
import { type ReactNode } from 'react';

vi.mock('@nemo/common/src/providers/toast/useToast');
vi.mock('@nemo/sdk/generated/platform/files');
vi.mock('@nemo/sdk/generated/platform/model-deployment-configs');
vi.mock('@nemo/sdk/generated/platform/model-deployments');
vi.mock('@nemo/sdk/generated/platform/models');

const mockUseToast = vi.mocked(useToast);
const mockFilesCreateFileset = vi.mocked(filesCreateFileset);
const mockModelsCreateModel = vi.mocked(modelsCreateModel);
const mockModelsGetModel = vi.mocked(modelsGetModel);
const mockModelsCreateDeploymentConfig = vi.mocked(modelsCreateDeploymentConfig);
const mockModelsGetLatestDeploymentConfig = vi.mocked(modelsGetLatestDeploymentConfig);
const mockModelsCreateDeployment = vi.mocked(modelsCreateDeployment);

const workspace = 'ws';

function wrapper({ children }: { children: ReactNode }) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

function baseWorkspaceValues(overrides: Partial<WizardFormValues> = {}): WizardFormValues {
  return {
    ...defaultWizardValues(),
    source: SOURCE_WORKSPACE,
    name: 'my-deploy',
    gpu: 2,
    ...overrides,
  };
}

describe('useCreateDeploymentBySource — workspace source', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseToast.mockReturnValue({
      success: vi.fn(),
      error: vi.fn(),
      info: vi.fn(),
      warning: vi.fn(),
      workingWithId: vi.fn(),
      dismissToast: vi.fn(),
    } as unknown as ReturnType<typeof useToast>);

    mockFilesCreateFileset.mockResolvedValue(undefined as never);
    mockModelsCreateModel.mockResolvedValue(undefined as never);
    mockModelsCreateDeploymentConfig.mockResolvedValue(undefined as never);
    mockModelsCreateDeployment.mockResolvedValue(undefined as never);
  });

  it('deploys an existing model entity without creating a fileset or model', async () => {
    const { result } = renderHook(() => useCreateDeploymentBySource(workspace), { wrapper });
    const onSuccess = vi.fn();

    await act(async () => {
      await result.current.createDeploymentFromWizard(
        baseWorkspaceValues({
          workspacePickerType: WORKSPACE_PICKER_MODEL,
          modelRef: 'other-ws/existing-model',
        }),
        onSuccess
      );
    });

    expect(mockFilesCreateFileset).not.toHaveBeenCalled();
    expect(mockModelsCreateModel).not.toHaveBeenCalled();

    expect(mockModelsCreateDeploymentConfig).toHaveBeenCalledWith(workspace, {
      name: 'my-deploy-config',
      engine: 'vllm',
      model_spec: {
        model_namespace: 'other-ws',
        model_name: 'existing-model',
        lora_enabled: true,
      },
      executor_config: {
        gpu: 2,
      },
      model_entity_id: 'other-ws/existing-model',
    });

    expect(mockModelsCreateDeployment).toHaveBeenCalledWith(workspace, {
      name: 'my-deploy-deployment',
      config: 'my-deploy-config',
    });

    expect(onSuccess).toHaveBeenCalled();
  });

  it('registers a model entity from the selected fileset before deploying', async () => {
    const { result } = renderHook(() => useCreateDeploymentBySource(workspace), { wrapper });
    const onSuccess = vi.fn();

    await act(async () => {
      await result.current.createDeploymentFromWizard(
        baseWorkspaceValues({
          workspacePickerType: WORKSPACE_PICKER_FILESET,
          fileset: 'other-ws/some-fileset',
        }),
        onSuccess
      );
    });

    expect(mockFilesCreateFileset).not.toHaveBeenCalled();

    expect(mockModelsCreateModel).toHaveBeenCalledWith(workspace, {
      name: 'my-deploy',
      fileset: 'other-ws/some-fileset',
    });

    expect(mockModelsCreateDeploymentConfig).toHaveBeenCalledWith(workspace, {
      name: 'my-deploy-config',
      engine: 'vllm',
      model_spec: {
        model_namespace: workspace,
        model_name: 'my-deploy',
        lora_enabled: true,
      },
      executor_config: {
        gpu: 2,
      },
      model_entity_id: 'ws/my-deploy',
    });

    expect(mockModelsCreateDeployment).toHaveBeenCalledWith(workspace, {
      name: 'my-deploy-deployment',
      config: 'my-deploy-config',
    });

    expect(onSuccess).toHaveBeenCalled();
  });

  it('sends the selected engine and image overrides', async () => {
    const { result } = renderHook(() => useCreateDeploymentBySource(workspace), { wrapper });

    await act(async () => {
      await result.current.createDeploymentFromWizard(
        baseWorkspaceValues({
          workspacePickerType: WORKSPACE_PICKER_MODEL,
          modelRef: 'ws/m',
          engine: Engine.nim,
          imageName: '  nvcr.io/nim/meta/llama-3.1-8b-instruct  ',
          imageTag: ' 1.8.5 ',
        }),
        vi.fn()
      );
    });

    expect(mockModelsCreateDeploymentConfig).toHaveBeenCalledWith(
      workspace,
      expect.objectContaining({
        engine: 'nim',
        executor_config: {
          gpu: 2,
          image_name: 'nvcr.io/nim/meta/llama-3.1-8b-instruct',
          image_tag: '1.8.5',
        },
      })
    );
  });

  it('omits blank image fields so the engine default is used', async () => {
    const { result } = renderHook(() => useCreateDeploymentBySource(workspace), { wrapper });

    await act(async () => {
      await result.current.createDeploymentFromWizard(
        baseWorkspaceValues({
          workspacePickerType: WORKSPACE_PICKER_MODEL,
          modelRef: 'ws/m',
          engine: Engine.vllm,
          imageName: '   ',
          imageTag: '',
        }),
        vi.fn()
      );
    });

    const [, request] = mockModelsCreateDeploymentConfig.mock.calls.at(-1)!;
    expect(request.executor_config).toEqual({ gpu: 2 });
    expect(request.executor_config).not.toHaveProperty('image_name');
  });

  it('surfaces submit errors and does not call onSuccess', async () => {
    mockModelsCreateDeploymentConfig.mockRejectedValueOnce(new Error('boom'));

    const { result } = renderHook(() => useCreateDeploymentBySource(workspace), { wrapper });
    const onSuccess = vi.fn();

    await act(async () => {
      await result.current.createDeploymentFromWizard(
        baseWorkspaceValues({
          workspacePickerType: WORKSPACE_PICKER_MODEL,
          modelRef: 'ws/m',
        }),
        onSuccess
      );
    });

    expect(onSuccess).not.toHaveBeenCalled();
    expect(result.current.submitError).toBeTruthy();
  });
});

/** An API rejection shaped like the Axios errors the SDK throws. */
function httpError(status: number) {
  return Object.assign(new Error(`request failed with status ${status}`), {
    response: { status },
  });
}

describe('useCreateDeploymentBySource — huggingface name collisions', () => {
  function huggingFaceValues(overrides: Partial<WizardFormValues> = {}): WizardFormValues {
    return {
      ...defaultWizardValues(),
      source: SOURCE_HF,
      name: 'qwen-qwen2.5-7b-instruct',
      repoId: 'Qwen/Qwen2.5-7B-Instruct',
      ...overrides,
    };
  }

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseToast.mockReturnValue({
      success: vi.fn(),
      error: vi.fn(),
      info: vi.fn(),
      warning: vi.fn(),
      workingWithId: vi.fn(),
      dismissToast: vi.fn(),
    } as unknown as ReturnType<typeof useToast>);

    mockFilesCreateFileset.mockResolvedValue(undefined as never);
    mockModelsCreateModel.mockResolvedValue(undefined as never);
    mockModelsCreateDeploymentConfig.mockResolvedValue(undefined as never);
    mockModelsCreateDeployment.mockResolvedValue(undefined as never);
    // Default: the name is free, which the API signals with a 404.
    mockModelsGetModel.mockRejectedValue(httpError(404));
  });

  it('fails before creating anything when the model name is taken', async () => {
    mockModelsGetModel.mockResolvedValue({ name: 'qwen-qwen2.5-7b-instruct' } as never);

    const { result } = renderHook(() => useCreateDeploymentBySource(workspace), { wrapper });
    const onSuccess = vi.fn();

    await act(async () => {
      await result.current.createDeploymentFromWizard(huggingFaceValues(), onSuccess);
    });

    // No fileset is created, so nothing is left orphaned.
    expect(mockFilesCreateFileset).not.toHaveBeenCalled();
    expect(mockModelsCreateModel).not.toHaveBeenCalled();
    expect(onSuccess).not.toHaveBeenCalled();
    expect(result.current.submitError).toContain('qwen-qwen2.5-7b-instruct');
    expect(result.current.submitError).toContain('already exists');
  });

  it('proceeds normally when the name is free', async () => {
    const { result } = renderHook(() => useCreateDeploymentBySource(workspace), { wrapper });

    await act(async () => {
      await result.current.createDeploymentFromWizard(huggingFaceValues(), vi.fn());
    });

    expect(mockFilesCreateFileset).toHaveBeenCalledTimes(1);
    expect(mockModelsCreateModel).toHaveBeenCalledTimes(1);
    expect(result.current.submitError).toBeNull();
  });

  it('fails without creating anything when the lookup fails for a reason other than 404', async () => {
    mockModelsGetModel.mockRejectedValue(httpError(503));

    const { result } = renderHook(() => useCreateDeploymentBySource(workspace), { wrapper });
    const onSuccess = vi.fn();

    await act(async () => {
      await result.current.createDeploymentFromWizard(huggingFaceValues(), onSuccess);
    });

    // A failed lookup says nothing about the name, so nothing is created.
    expect(mockFilesCreateFileset).not.toHaveBeenCalled();
    expect(mockModelsCreateModel).not.toHaveBeenCalled();
    expect(mockModelsCreateDeploymentConfig).not.toHaveBeenCalled();
    expect(mockModelsCreateDeployment).not.toHaveBeenCalled();
    expect(onSuccess).not.toHaveBeenCalled();
    expect(result.current.submitError).toBeTruthy();
  });
});

describe('ensureWorkspaceDeploymentConfig', () => {
  const noop = () => {};

  /** What axios raises on a 409; `isVersionConflictError` tests `instanceof AxiosError`. */
  const conflict = () =>
    new AxiosError('Conflict', '409', undefined, undefined, {
      status: 409,
      data: {},
      statusText: 'Conflict',
      headers: {},
      config: { headers: new AxiosHeaders() },
    });

  beforeEach(() => {
    vi.clearAllMocks();
    mockModelsGetModel.mockRejectedValue({ response: { status: 404 } });
  });

  it('reports a freshly created config as not reused', async () => {
    const created = { name: 'base-config', engine: Engine.vllm };
    mockModelsCreateDeploymentConfig.mockResolvedValue(
      created as Awaited<ReturnType<typeof modelsCreateDeploymentConfig>>
    );

    const result = await ensureWorkspaceDeploymentConfig(
      workspace,
      baseWorkspaceValues({ workspacePickerType: WORKSPACE_PICKER_MODEL, modelRef: 'ws/base' }),
      'base-config',
      noop
    );

    expect(result.reused).toBe(false);
    expect(mockModelsGetLatestDeploymentConfig).not.toHaveBeenCalled();
  });

  // The reported bug: a second adapter run against the same base reuses the name while
  // the first job is still training, so the 409 blocked the job from ever starting.
  it('adopts the existing config on a 409 instead of failing', async () => {
    const existing = { name: 'base-config', engine: Engine.nim };
    mockModelsCreateDeploymentConfig.mockRejectedValue(conflict());
    mockModelsGetLatestDeploymentConfig.mockResolvedValue(
      existing as Awaited<ReturnType<typeof modelsGetLatestDeploymentConfig>>
    );

    const result = await ensureWorkspaceDeploymentConfig(
      workspace,
      baseWorkspaceValues({ workspacePickerType: WORKSPACE_PICKER_MODEL, modelRef: 'ws/base' }),
      'base-config',
      noop
    );

    expect(result.reused).toBe(true);
    // The adopted config is returned, not the values that were submitted — the caller
    // needs the real settings to be able to report them.
    expect(result.config).toBe(existing);
    expect(mockModelsGetLatestDeploymentConfig).toHaveBeenCalledWith(workspace, 'base-config');
  });

  // Only a conflict means "already there". Anything else is a real failure and must
  // not be quietly converted into a reused config.
  it('propagates a non-conflict failure', async () => {
    mockModelsCreateDeploymentConfig.mockRejectedValue(new Error('image pull denied'));

    await expect(
      ensureWorkspaceDeploymentConfig(
        workspace,
        baseWorkspaceValues({ workspacePickerType: WORKSPACE_PICKER_MODEL, modelRef: 'ws/base' }),
        'base-config',
        noop
      )
    ).rejects.toThrow('image pull denied');
    expect(mockModelsGetLatestDeploymentConfig).not.toHaveBeenCalled();
  });
});

describe('createUnboundDeploymentConfig', () => {
  const noop = () => {};

  beforeEach(() => {
    vi.clearAllMocks();
    mockModelsCreateDeploymentConfig.mockResolvedValue({ name: 'out-config' } as Awaited<
      ReturnType<typeof modelsCreateDeploymentConfig>
    >);
  });

  // The whole point of the unbound shape: `is_unbound_deployment_config` returns true
  // only when *neither* link to a model is set, and a config that names a model the
  // run has not produced yet is what the compilers used to reject.
  it('sends neither model_entity_id nor a model name', async () => {
    await createUnboundDeploymentConfig(
      workspace,
      baseWorkspaceValues({ workspacePickerType: WORKSPACE_PICKER_MODEL, modelRef: 'ws/output' }),
      'out-config',
      noop
    );

    expect(mockModelsCreateDeploymentConfig).toHaveBeenCalledTimes(1);
    const body = mockModelsCreateDeploymentConfig.mock.calls[0][1];
    expect(body.model_entity_id).toBeUndefined();
    expect(body.model_spec.model_name).toBeUndefined();
    expect(body.model_spec.model_namespace).toBeUndefined();
  });

  // A serving option rather than a model link: the task copies model_spec onto the
  // derived config and only overwrites the name/namespace, so this choice survives.
  it('still sends lora_enabled, the engine and the executor', async () => {
    await createUnboundDeploymentConfig(
      workspace,
      baseWorkspaceValues({
        workspacePickerType: WORKSPACE_PICKER_MODEL,
        modelRef: 'ws/output',
        engine: Engine.vllm,
        loraEnabled: true,
      }),
      'out-config',
      noop
    );

    expect(mockModelsCreateDeploymentConfig).toHaveBeenCalledWith(
      workspace,
      expect.objectContaining({
        name: 'out-config',
        engine: Engine.vllm,
        model_spec: { lora_enabled: true },
        executor_config: expect.objectContaining({ gpu: 2 }),
      })
    );
  });

  // No model means no Model Entity to register, unlike the fileset branch of the
  // bound creator — an unbound config touches nothing but itself.
  it('registers no model', async () => {
    await createUnboundDeploymentConfig(
      workspace,
      baseWorkspaceValues({ workspacePickerType: WORKSPACE_PICKER_MODEL, modelRef: 'ws/output' }),
      'out-config',
      noop
    );

    expect(mockModelsCreateModel).not.toHaveBeenCalled();
  });

  it('adopts the existing config on a 409', async () => {
    const existing = { name: 'out-config', engine: Engine.nim };
    mockModelsCreateDeploymentConfig.mockRejectedValue(
      new AxiosError('Conflict', '409', undefined, undefined, {
        status: 409,
        data: {},
        statusText: 'Conflict',
        headers: {},
        config: { headers: new AxiosHeaders() },
      })
    );
    mockModelsGetLatestDeploymentConfig.mockResolvedValue(
      existing as Awaited<ReturnType<typeof modelsGetLatestDeploymentConfig>>
    );

    const result = await ensureUnboundDeploymentConfig(
      workspace,
      baseWorkspaceValues({ workspacePickerType: WORKSPACE_PICKER_MODEL, modelRef: 'ws/output' }),
      'out-config',
      noop
    );

    expect(result).toEqual({ config: existing, reused: true });
  });
});
