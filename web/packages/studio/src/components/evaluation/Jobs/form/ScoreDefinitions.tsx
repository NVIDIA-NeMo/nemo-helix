// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { InputErrorText } from '@nemo/common/src/components/InputErrorText';
import { Badge, Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import {
  ScoreModal,
  type MetricPanelFormData,
  type PanelScoreFormData,
} from '@studio/components/evaluation/Jobs/form/ScoreModal';
import { Pencil, Plus, Trash } from 'lucide-react';
import { FC, useState } from 'react';
import { FieldError, useFormContext, useWatch } from 'react-hook-form';

type ModalState = { mode: 'closed' } | { mode: 'add' } | { mode: 'edit'; index: number };

export const ScoreDefinitions: FC = () => {
  const {
    control,
    setValue,
    formState: { errors },
  } = useFormContext<MetricPanelFormData>();
  const scores = useWatch({ control, name: 'body.scores' });
  const scoresError = (errors.body?.scores as FieldError | undefined)?.message;
  const [modalState, setModalState] = useState<ModalState>({ mode: 'closed' });

  const handleAdd = (score: PanelScoreFormData) => {
    setValue('body.scores', [...scores, score], { shouldValidate: true });
  };

  const handleEdit = (score: PanelScoreFormData, index: number) => {
    const updated = [...scores];
    updated[index] = score;
    setValue('body.scores', updated, { shouldValidate: true });
  };

  const handleDelete = (index: number) => {
    setValue(
      'body.scores',
      scores.filter((_, i) => i !== index),
      { shouldValidate: true }
    );
  };

  const handleSave = (score: PanelScoreFormData) => {
    if (modalState.mode === 'edit') {
      handleEdit(score, modalState.index);
    } else {
      handleAdd(score);
    }
  };

  return (
    <Stack gap="density-sm">
      {scores.length === 0 ? (
        <Button
          kind="secondary"
          type="button"
          className="w-full rounded-lg border border-dashed border-base p-4 flex items-center justify-center transition-colors cursor-pointer"
          onClick={() => setModalState({ mode: 'add' })}
        >
          <Flex gap="density-xs" align="center">
            <Plus className="size-3.5" aria-hidden />
            <Text kind="body/regular/sm">Add Score</Text>
          </Flex>
        </Button>
      ) : (
        <>
          {scores.map((score, index) => (
            <Stack
              key={index}
              gap="density-xs"
              className="border border-base rounded-lg p-4 bg-surface-raised"
            >
              <Flex gap="density-sm" align="center" justify="between">
                <Flex gap="density-sm" align="center" className="min-w-0">
                  <Text kind="body/bold/md" className="shrink-0">
                    {score.name}
                  </Text>
                  {score.description && (
                    <Text
                      kind="body/regular/md"
                      className="truncate min-w-0 text-content-secondary"
                    >
                      {score.description}
                    </Text>
                  )}
                </Flex>
                <Flex gap="density-xs" className="shrink-0">
                  <Button
                    type="button"
                    kind="tertiary"
                    size="small"
                    aria-label={`Edit score ${score.name}`}
                    onClick={() => setModalState({ mode: 'edit', index })}
                  >
                    <Pencil className="size-3.5" aria-hidden />
                  </Button>
                  <Button
                    type="button"
                    kind="tertiary"
                    size="small"
                    aria-label={`Delete score ${score.name}`}
                    onClick={() => handleDelete(index)}
                  >
                    <Trash className="size-3.5" aria-hidden />
                  </Button>
                </Flex>
              </Flex>

              <Flex gap="density-xs" align="center" wrap="wrap">
                <Text kind="label/bold/sm" className="shrink-0">
                  {score.scoreType === 'range' ? 'Range' : 'Rubric'}
                </Text>
                {score.scoreType === 'range' ? (
                  <Badge kind="solid" color="gray">
                    {score.minimum}–{score.maximum}
                  </Badge>
                ) : (
                  score.rubric.map((item, i) => (
                    <Badge key={i} kind="solid" color="gray" title={item.description ?? undefined}>
                      {/* `label: value` only when they differ. A numeric rubric
                          labels its levels by their own value, so repeating it
                          reads as "0: 0" and hides the level's meaning. */}
                      {String(item.label) === String(item.value)
                        ? item.label
                        : `${item.label}: ${item.value}`}
                    </Badge>
                  ))
                )}
              </Flex>
            </Stack>
          ))}

          <Button
            kind="tertiary"
            size="small"
            type="button"
            onClick={() => setModalState({ mode: 'add' })}
          >
            <Plus className="size-3.5" aria-hidden />
            Add Score
          </Button>
        </>
      )}

      {scoresError && <InputErrorText kind="body/regular/sm">{scoresError}</InputErrorText>}

      <ScoreModal
        open={modalState.mode !== 'closed'}
        onClose={() => setModalState({ mode: 'closed' })}
        onSave={handleSave}
        initialValues={modalState.mode === 'edit' ? scores[modalState.index] : undefined}
      />
    </Stack>
  );
};
