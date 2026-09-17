// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  defaultWizardValues,
  SOURCE_HF,
  SOURCE_NGC,
  type WizardFormValues,
} from '@studio/routes/NewDeploymentRoute/schema';
import { useHuggingFaceNameDefault } from '@studio/routes/NewDeploymentRoute/useHuggingFaceNameDefault';
import { act, renderHook } from '@testing-library/react';
import { useForm } from 'react-hook-form';

function setup(source: WizardFormValues['source']) {
  return renderHook(() => {
    const form = useForm<WizardFormValues>({
      defaultValues: { ...defaultWizardValues(), source, name: 'seeded-default-name' },
    });
    useHuggingFaceNameDefault(form.control, form.setValue);
    return form;
  });
}

describe('useHuggingFaceNameDefault', () => {
  it('derives the name from the repo id on the HuggingFace source', async () => {
    const { result } = setup(SOURCE_HF);

    await act(async () => {
      result.current.setValue('repoId', 'Qwen/Qwen2.5-7B-Instruct', { shouldDirty: true });
    });

    expect(result.current.getValues('name')).toBe('qwen-qwen2.5-7b-instruct');
  });

  it('tracks further repo id edits while the name is untouched', async () => {
    const { result } = setup(SOURCE_HF);

    await act(async () => {
      result.current.setValue('repoId', 'Qwen/Qwen2.5-7B-Instruct', { shouldDirty: true });
    });
    await act(async () => {
      result.current.setValue('repoId', 'deepseek-ai/DeepSeek-V4-Flash', { shouldDirty: true });
    });

    expect(result.current.getValues('name')).toBe('deepseek-ai-deepseek-v4-flash');
  });

  it('does not mark the name field dirty when deriving', async () => {
    const { result } = setup(SOURCE_HF);

    await act(async () => {
      result.current.setValue('repoId', 'Qwen/Qwen2.5-7B-Instruct', { shouldDirty: true });
    });

    expect(result.current.formState.dirtyFields.name).toBeFalsy();
  });

  it('stops deriving once the user edits the name', async () => {
    const { result } = setup(SOURCE_HF);

    await act(async () => {
      result.current.setValue('name', 'my-own-name', { shouldDirty: true });
    });
    await act(async () => {
      result.current.setValue('repoId', 'Qwen/Qwen2.5-7B-Instruct', { shouldDirty: true });
    });

    expect(result.current.getValues('name')).toBe('my-own-name');
  });

  it('keeps the name after the user edits it and then restores the default', async () => {
    const { result } = setup(SOURCE_HF);

    await act(async () => {
      result.current.setValue('name', 'my-own-name', { shouldDirty: true });
    });
    // Undoing back to the generated default clears RHF's value-based dirty flag,
    // but the user has still taken ownership of the field.
    await act(async () => {
      result.current.setValue('name', 'seeded-default-name', { shouldDirty: true });
    });
    await act(async () => {
      result.current.setValue('repoId', 'Qwen/Qwen2.5-7B-Instruct', { shouldDirty: true });
    });

    expect(result.current.getValues('name')).toBe('seeded-default-name');
  });

  it('resumes deriving after the form is reset for a fresh wizard', async () => {
    const { result } = setup(SOURCE_HF);

    await act(async () => {
      result.current.setValue('name', 'my-own-name', { shouldDirty: true });
    });
    await act(async () => {
      result.current.reset({
        ...defaultWizardValues(),
        source: SOURCE_HF,
        name: 'a-new-default-name',
      });
    });
    await act(async () => {
      result.current.setValue('repoId', 'Qwen/Qwen2.5-7B-Instruct', { shouldDirty: true });
    });

    expect(result.current.getValues('name')).toBe('qwen-qwen2.5-7b-instruct');
  });

  it('leaves the existing default alone for an unusable repo id', async () => {
    const { result } = setup(SOURCE_HF);

    await act(async () => {
      result.current.setValue('repoId', '   ', { shouldDirty: true });
    });

    expect(result.current.getValues('name')).toBe('seeded-default-name');
  });

  it('does nothing on a non-HuggingFace source', async () => {
    const { result } = setup(SOURCE_NGC);

    await act(async () => {
      result.current.setValue('repoId', 'Qwen/Qwen2.5-7B-Instruct', { shouldDirty: true });
    });

    expect(result.current.getValues('name')).toBe('seeded-default-name');
  });
});
