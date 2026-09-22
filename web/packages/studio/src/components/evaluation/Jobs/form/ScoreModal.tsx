// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { Button, Flex, SegmentedControl, Stack, Text } from '@nvidia/foundations-react-core';
import { Trash } from 'lucide-react';
import { FC, useEffect, useRef } from 'react';
import { useFieldArray, useForm } from 'react-hook-form';
import { z } from 'zod';

/** Coerces to number but treats empty strings as undefined so required checks work. */
const requiredNumber = (message: string) =>
  z.preprocess(
    (val) => (val === '' || val === undefined ? undefined : Number(val)),
    z.number({ required_error: message })
  );

const scoreNameSchema = z
  .string()
  .min(1, 'Score name is required')
  .regex(/^[a-z0-9_]+$/, 'Only lowercase letters, numbers, and underscores allowed');

const localRangeSchema = z
  .object({
    scoreType: z.literal('range'),
    name: scoreNameSchema,
    description: z.string().optional(),
    minimum: requiredNumber('Minimum value is required'),
    maximum: requiredNumber('Maximum value is required'),
  })
  .refine((data) => data.minimum < data.maximum, {
    message: 'Minimum must be less than maximum',
    path: ['minimum'],
  });

const localRubricSchema = z.object({
  scoreType: z.literal('rubric'),
  name: scoreNameSchema,
  description: z.string().optional(),
  rubric: z
    .array(
      z.object({
        label: z.string().min(1, 'Label is required'),
        description: z.string().optional(),
        value: requiredNumber('Required'),
      })
    )
    .min(2, 'At least two rubric items are required'),
});

const localScoreSchema = z.union([localRangeSchema, localRubricSchema]);

type LocalPanelScoreFormData = z.infer<typeof localScoreSchema>;
/** Public alias used by ScoreDefinitions and other consumers. */
export type PanelScoreFormData = LocalPanelScoreFormData;
/** Minimal form shape for components that host a score list. */
export interface MetricPanelFormData {
  body: {
    scores: PanelScoreFormData[];
    [key: string]: unknown;
  };
  [key: string]: unknown;
}
type LocalRangeScoreFormData = z.infer<typeof localRangeSchema>;
type LocalRubricScoreFormData = z.infer<typeof localRubricSchema>;
type ScoreType = LocalPanelScoreFormData['scoreType'];
type ScoreCommonFields = Pick<LocalPanelScoreFormData, 'name' | 'description'>;

interface CachedScoreValues {
  range: LocalRangeScoreFormData;
  rubric: LocalRubricScoreFormData;
}

const DEFAULT_RANGE: LocalRangeScoreFormData = {
  scoreType: 'range',
  name: '',
  description: '',
  minimum: 1,
  maximum: 5,
};

const DEFAULT_RUBRIC: LocalRubricScoreFormData = {
  scoreType: 'rubric',
  name: '',
  description: '',
  rubric: [
    { label: '', description: '', value: '' as unknown as number },
    { label: '', description: '', value: '' as unknown as number },
  ],
};

const isScoreType = (value: string): value is ScoreType => value === 'range' || value === 'rubric';

const withCommonFields = <T extends LocalPanelScoreFormData>(
  score: T,
  commonFields: ScoreCommonFields
): T => ({ ...score, ...commonFields });

const getCommonFields = (score: LocalPanelScoreFormData): ScoreCommonFields => ({
  name: score.name,
  description: score.description,
});

const getCachedScoreValues = (score?: PanelScoreFormData): CachedScoreValues => {
  const range = structuredClone(DEFAULT_RANGE);
  const rubric = structuredClone(DEFAULT_RUBRIC);

  if (!score) {
    return { range, rubric };
  }

  const commonFields = getCommonFields(score);

  if (score.scoreType === 'range') {
    return {
      range: structuredClone(score),
      rubric: withCommonFields(rubric, commonFields),
    };
  }

  return {
    range: withCommonFields(range, commonFields),
    rubric: structuredClone(score),
  };
};

interface ScoreModalProps {
  open: boolean;
  onClose: () => void;
  onSave: (score: PanelScoreFormData) => void;
  initialValues?: PanelScoreFormData;
}

export const ScoreModal: FC<ScoreModalProps> = ({ open, onClose, onSave, initialValues }) => {
  const cachedScoreValues = useRef<CachedScoreValues>(getCachedScoreValues(initialValues));
  const form = useForm<LocalPanelScoreFormData>({
    resolver: zodResolver(localScoreSchema),
    defaultValues: initialValues ?? cachedScoreValues.current.rubric,
  });

  const scoreType = form.watch('scoreType');

  const rubricFields = useFieldArray({
    control: form.control,
    name: 'rubric',
  });

  useEffect(() => {
    if (open) {
      cachedScoreValues.current = getCachedScoreValues(initialValues);
      form.reset(initialValues ?? cachedScoreValues.current.rubric);
    }
  }, [open, initialValues, form]);

  const handleTypeChange = (newType: string) => {
    if (!isScoreType(newType) || newType === scoreType) {
      return;
    }

    const currentScore = form.getValues();
    const commonFields = getCommonFields(currentScore);

    if (currentScore.scoreType === 'range') {
      cachedScoreValues.current.range = currentScore;
    } else {
      cachedScoreValues.current.rubric = currentScore;
    }

    if (newType === 'range') {
      const nextScore = withCommonFields(cachedScoreValues.current.range, commonFields);
      cachedScoreValues.current.range = nextScore;
      form.reset(nextScore);
    } else {
      const nextScore = withCommonFields(cachedScoreValues.current.rubric, commonFields);
      cachedScoreValues.current.rubric = nextScore;
      form.reset(nextScore);
    }
  };

  const handleSubmit = (data: LocalPanelScoreFormData) => {
    onSave(data as PanelScoreFormData);
    onClose();
  };

  return (
    <FormModal
      open={open}
      title={initialValues ? 'Edit Score' : 'Add Score'}
      submitButtonText={initialValues ? 'Edit Score' : 'Add Score'}
      onClose={onClose}
      onSubmit={form.handleSubmit(handleSubmit)}
      className="w-[560px]"
    >
      <Stack gap="density-lg">
        <SegmentedControl
          className="w-full"
          value={scoreType}
          onValueChange={handleTypeChange}
          items={[
            { value: 'rubric', children: 'Rubric' },
            { value: 'range', children: 'Range' },
          ]}
        />

        <ControlledTextInput
          useControllerProps={{ control: form.control, name: 'name' }}
          label="Name"
          placeholder="e.g., quality"
        />

        <ControlledTextInput
          useControllerProps={{ control: form.control, name: 'description' }}
          label="Description (optional)"
        />

        {scoreType === 'range' && (
          <Flex gap="density-lg">
            <ControlledTextInput
              useControllerProps={{ control: form.control, name: 'minimum' }}
              label="Minimum"
              type="number"
            />
            <ControlledTextInput
              useControllerProps={{ control: form.control, name: 'maximum' }}
              label="Maximum"
              type="number"
            />
          </Flex>
        )}

        {scoreType === 'rubric' && (
          <Stack gap="density-sm">
            <Flex gap="density-sm">
              <Text className="flex-1" kind="label/bold/sm">
                Label
              </Text>
              <Text className="flex-1" kind="label/bold/sm">
                Description
              </Text>
              <Text className="w-[80px] shrink-0" kind="label/bold/sm">
                Value
              </Text>
              <div className="w-[36px] shrink-0" />
            </Flex>
            {rubricFields.fields.map((field, index) => (
              <Flex key={field.id} gap="density-sm" align="start">
                <div className="flex-1">
                  <ControlledTextInput
                    useControllerProps={{ control: form.control, name: `rubric.${index}.label` }}
                    placeholder="Label"
                  />
                </div>
                <div className="flex-1">
                  <ControlledTextInput
                    useControllerProps={{
                      control: form.control,
                      name: `rubric.${index}.description`,
                    }}
                    placeholder="Description"
                  />
                </div>
                <div className="w-[80px] shrink-0">
                  <ControlledTextInput
                    useControllerProps={{ control: form.control, name: `rubric.${index}.value` }}
                    type="number"
                  />
                </div>
                <Button
                  kind="tertiary"
                  aria-label="Remove rubric item"
                  onClick={() => rubricFields.remove(index)}
                  disabled={rubricFields.fields.length <= 2}
                  type="button"
                >
                  <Trash />
                </Button>
              </Flex>
            ))}
            <Button
              kind="secondary"
              type="button"
              onClick={() =>
                rubricFields.append({ label: '', description: '', value: '' as unknown as number })
              }
            >
              Add Item
            </Button>
          </Stack>
        )}
      </Stack>
    </FormModal>
  );
};
