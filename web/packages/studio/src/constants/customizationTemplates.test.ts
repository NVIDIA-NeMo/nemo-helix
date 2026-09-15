// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CUSTOMIZATION_TEMPLATES } from '@studio/constants/customizationTemplates';
import { customizationFormSchema, formToAutomodelCreate } from '@studio/util/forms/customization';

const WORKSPACE = 'my-workspace';
const DATASET_REF = 'my-workspace/sft-dataset';

describe('CUSTOMIZATION_TEMPLATES', () => {
  it('exposes unique template ids', () => {
    const ids = CUSTOMIZATION_TEMPLATES.map((t) => t.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  describe.each(CUSTOMIZATION_TEMPLATES.map((t) => [t.id, t] as const))('%s', (_id, template) => {
    const fields = template.buildFormSpec(WORKSPACE, DATASET_REF);

    it('builds form fields that pass the customization schema', () => {
      const result = customizationFormSchema.safeParse(fields);
      expect(result.success).toBe(true);
    });

    it('namespaces the base model with the workspace', () => {
      expect(fields.automodel.model.startsWith(`${WORKSPACE}/`)).toBe(true);
    });

    it('wires the created dataset into training and validation', () => {
      expect(fields.automodel.dataset.training).toBe(DATASET_REF);
      expect(fields.automodel.dataset.validation).toBe(DATASET_REF);
    });

    it('registers every model it references', () => {
      const registered = template.models.map((m) => `${WORKSPACE}/${m.name}`);
      expect(registered).toContain(fields.automodel.model);
      const teacher = fields.automodel.training.teacher_model;
      if (teacher) expect(registered).toContain(teacher);
    });
  });

  describe('lora-nemotron-3-super-text2sql', () => {
    const template = CUSTOMIZATION_TEMPLATES.find(
      (t) => t.id === 'lora-nemotron-3-super-text2sql'
    )!;
    const fields = template.buildFormSpec(WORKSPACE, DATASET_REF);
    const { automodel } = fields;

    it('needs trust_remote_code but no HF token — the Nemotron repo is not gated', () => {
      // Unlike the Llama templates, this checkpoint is public. Marking it gated would
      // make provisioning fail for anyone without an `hf-token` secret configured.
      expect(template.models).toHaveLength(1);
      expect(template.models[0].hfRepoId).toBe('nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16');
      expect(template.models[0].requiresHfToken).toBe(false);
      expect(template.models[0].trustRemoteCode).toBe(true);
    });

    it('spreads the MoE experts across all 8 GPUs', () => {
      // The point of the recipe: without ep_size matching the GPU count, a 120B A12B
      // MoE will not fit on a single 8x80GB node.
      expect(automodel.parallelism?.expert_parallel_size).toBe(8);
      expect(automodel.parallelism?.num_gpus_per_node).toBe(8);
      expect(automodel.parallelism?.num_nodes).toBe(1);
      expect(automodel.parallelism?.tensor_parallel_size).toBe(1);
    });

    it('excludes out_proj from LoRA, where Nemotron mamba kernels bypass adapters', () => {
      expect(automodel.training.lora?.exclude_modules).toEqual(['*.out_proj']);
    });

    it('mirrors the cookbook LoRA and optimizer settings', () => {
      expect(automodel.training.lora?.rank).toBe(8);
      expect(automodel.training.lora?.alpha).toBe(32);
      expect(automodel.training.lora?.use_triton).toBe(true);
      expect(automodel.training.max_seq_length).toBe(4096);
      expect(automodel.training.precision).toBe('bf16');
      expect(automodel.optimizer?.learning_rate).toBe(1e-5);
      expect(automodel.optimizer?.weight_decay).toBe(0);
      expect(automodel.optimizer?.optimizer).toBe('Adam');
      expect(automodel.optimizer?.lr_decay_style).toBe('cosine');
      expect(automodel.batch?.global_batch_size).toBe(8);
      expect(automodel.batch?.micro_batch_size).toBe(1);
      expect(automodel.schedule?.epochs).toBe(1);
    });

    it('survives the round trip into a create-job request', () => {
      // exclude_modules and expert_parallel_size are optional passthrough fields; a
      // recipe that set them only to have the mapper drop them would silently OOM.
      const spec = formToAutomodelCreate(fields).spec;
      expect(spec.training.lora?.exclude_modules).toEqual(['*.out_proj']);
      expect(spec.parallelism?.expert_parallel_size).toBe(8);
    });
  });

  describe('Nemotron MoE recipes', () => {
    const MOE_IDS = [
      'lora-nemotron-3-super-text2sql',
      'lora-nemotron-3-ultra-text2sql',
      'lora-nemotron-35-lightning-text2sql',
    ];

    describe.each(MOE_IDS)('%s', (id) => {
      const template = CUSTOMIZATION_TEMPLATES.find((t) => t.id === id)!;
      const { automodel } = template.buildFormSpec(WORKSPACE, DATASET_REF);
      const p = automodel.parallelism!;

      it('satisfies the backend MoE parallelism constraints', () => {
        // services/automodel/.../backends/config.py raises for MoE models when
        // tensor_parallel_size > 1, or when expert_parallel_size is unset/<=1 on
        // multi-GPU. A recipe violating either fails the job at launch.
        const worldSize = p.num_nodes! * p.num_gpus_per_node!;
        const derivedDp =
          worldSize /
          (p.tensor_parallel_size! * p.pipeline_parallel_size! * p.context_parallel_size!);
        expect(p.tensor_parallel_size).toBe(1);
        expect(p.expert_parallel_size).toBeGreaterThan(1);
        // The rule is divisibility, not equality — ep need not span the whole world.
        expect((derivedDp * p.context_parallel_size!) % p.expert_parallel_size!).toBe(0);
      });

      it('keeps global batch divisible by micro batch x data parallel size', () => {
        // Same validator: gb % (mb * derived_dp) must be 0, else the job is rejected.
        const { automodel: a } = template.buildFormSpec(WORKSPACE, DATASET_REF);
        const derivedDp =
          (p.num_nodes! * p.num_gpus_per_node!) /
          (p.tensor_parallel_size! * p.pipeline_parallel_size! * p.context_parallel_size!);
        expect(a.batch!.global_batch_size! % (a.batch!.micro_batch_size! * derivedDp)).toBe(0);
      });

      it('excludes out_proj, which Nemotron mamba kernels bypass', () => {
        expect(automodel.training.lora?.exclude_modules).toEqual(['*.out_proj']);
      });

      it('needs trust_remote_code but no HF token — these repos are public', () => {
        expect(template.models.every((m) => m.trustRemoteCode)).toBe(true);
        expect(template.models.some((m) => m.requiresHfToken)).toBe(false);
      });

      it('keeps MoE-critical fields through the create-job mapping', () => {
        const spec = formToAutomodelCreate(template.buildFormSpec(WORKSPACE, DATASET_REF)).spec;
        expect(spec.parallelism?.expert_parallel_size).toBe(p.expert_parallel_size);
        expect(spec.training.lora?.exclude_modules).toEqual(['*.out_proj']);
      });
    });

    it('mirrors the Ultra cookbook topology and hyperparameters', () => {
      const { automodel } = CUSTOMIZATION_TEMPLATES.find(
        (t) => t.id === 'lora-nemotron-3-ultra-text2sql'
      )!.buildFormSpec(WORKSPACE, DATASET_REF);

      expect(automodel.parallelism?.num_nodes).toBe(4);
      expect(automodel.parallelism?.expert_parallel_size).toBe(32);
      expect(automodel.training.lora?.rank).toBe(32);
      expect(automodel.schedule?.max_steps).toBe(100);
      expect(automodel.batch?.global_batch_size).toBe(128);
      expect(automodel.batch?.micro_batch_size).toBe(4);
      expect(automodel.optimizer?.optimizer).toBe('AdamW');
      expect(automodel.optimizer?.learning_rate).toBe(1e-4);
      expect(automodel.optimizer?.min_learning_rate).toBe(1e-5);
      expect(automodel.optimizer?.weight_decay).toBe(0.1);
      expect(automodel.optimizer?.adam_beta2).toBe(0.95);
      expect(automodel.optimizer?.warmup_steps).toBe(10);
    });

    it('keeps Lightning on a single node, the smallest of the three', () => {
      const { automodel } = CUSTOMIZATION_TEMPLATES.find(
        (t) => t.id === 'lora-nemotron-35-lightning-text2sql'
      )!.buildFormSpec(WORKSPACE, DATASET_REF);

      expect(automodel.parallelism?.num_nodes).toBe(1);
      expect(automodel.parallelism?.expert_parallel_size).toBe(8);
      expect(automodel.training.lora?.rank).toBe(8);
      expect(automodel.batch?.global_batch_size).toBe(8);
      // packed_sequence_size: 0 in the cookbook.
      expect(automodel.batch?.sequence_packing).toBe(false);
    });
  });

  it('ships only Nemotron recipes, all LoRA on the automodel backend', () => {
    expect(CUSTOMIZATION_TEMPLATES).toHaveLength(3);
    for (const template of CUSTOMIZATION_TEMPLATES) {
      expect(template.trainingLabel).toBe('LoRA');
      const fields = template.buildFormSpec(WORKSPACE, DATASET_REF);
      expect(fields.backend).toBe('automodel');
      expect(fields.automodel.training.finetuning_type).toBe('lora');
      expect(fields.automodel.training.training_type).toBe('sft');
      // No distillation recipes remain, so no template should carry a teacher.
      expect(fields.automodel.training.teacher_model).toBeUndefined();
    }
  });
});

describe('template dataset converters', () => {
  describe('BIRD-SQL converter', () => {
    const birdDataset = CUSTOMIZATION_TEMPLATES.find(
      (t) => t.id === 'lora-nemotron-3-super-text2sql'
    )!.dataset;

    it('lays out schema, question and evidence as the cookbook prompt', () => {
      expect(
        birdDataset.convertRow({
          schema: 'CREATE TABLE t (id INT);',
          question: 'How many rows?',
          evidence: 'rows refers to COUNT(*)',
          SQL: 'SELECT COUNT(*) FROM t',
        })
      ).toEqual({
        prompt: 'CREATE TABLE t (id INT);\n\nHow many rows?\nrows refers to COUNT(*)',
        completion: 'SELECT COUNT(*) FROM t',
      });
    });

    it('keeps rows with empty evidence, which is common upstream', () => {
      const converted = birdDataset.convertRow({
        schema: 'CREATE TABLE t (id INT);',
        question: 'How many rows?',
        evidence: '',
        SQL: 'SELECT COUNT(*) FROM t',
      });
      expect(converted).not.toBeNull();
      expect(converted?.prompt).toBe('CREATE TABLE t (id INT);\n\nHow many rows?\n');
    });

    it.each([
      ['schema', { question: 'q', SQL: 'SELECT 1' }],
      ['question', { schema: 's', SQL: 'SELECT 1' }],
      ['SQL', { schema: 's', question: 'q' }],
    ])('drops rows missing %s', (_field, row) => {
      expect(birdDataset.convertRow(row)).toBeNull();
    });
  });
});

describe('template dataset sources', () => {
  /**
   * Two filesets per template: the read-only external one pointing at HuggingFace, and the
   * local one holding the converted JSONL. A name collision between them would make the
   * converted upload target the read-only fileset, which rejects writes.
   */
  it('never names a source fileset after a converted one', () => {
    const convertedNames = new Set(CUSTOMIZATION_TEMPLATES.map((t) => t.dataset.name));
    for (const { dataset } of CUSTOMIZATION_TEMPLATES) {
      expect(dataset.sourceFilesetName).not.toBe(dataset.name);
      expect(convertedNames.has(dataset.sourceFilesetName)).toBe(false);
    }
  });

  /** Sharing the name lets the second template's create collapse into a no-op 409. */
  it('reuses one source fileset per HuggingFace repo', () => {
    const byRepo = new Map<string, Set<string>>();
    for (const { dataset } of CUSTOMIZATION_TEMPLATES) {
      const names = byRepo.get(dataset.hfRepoId) ?? new Set<string>();
      names.add(dataset.sourceFilesetName);
      byRepo.set(dataset.hfRepoId, names);
    }
    for (const names of byRepo.values()) {
      expect(names.size).toBe(1);
    }
  });

  it.each([
    ['the current BIRD-SQL shard', 'data/train-00000-of-00001-fe8894d41b7815be.parquet', true],
    ['a re-sharded train split', 'data/train-00002-of-00004-abc123.parquet', true],
    ['an unsuffixed shard', 'train-00000-of-00001.parquet', true],
    ['the validation split', 'data/validation-00000-of-00001-abc.parquet', false],
    ['a non-Parquet file', 'data/train-00000-of-00001-abc.json', false],
    ['the readme', 'README.md', false],
    ['a differently-prefixed split', 'data/pretrain-00000-of-00001-abc.parquet', false],
  ])('matches %s: %s', (_label, path, expected) => {
    for (const { dataset } of CUSTOMIZATION_TEMPLATES) {
      expect(dataset.filePattern.test(path)).toBe(expected);
    }
  });
});
