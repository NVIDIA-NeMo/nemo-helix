<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# LLM-as-a-Judge Evaluation (CLI)

You have access to the `nemo` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nemo` CLI is available at `/app/.venv/bin/nemo`. Use `nemo --help` and subcommand `--help` flags to discover available commands and their options. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Context

- CLI authentication is pre-configured (already logged in)
- A workspace named `eval-judge-workspace` is pre-configured
- An inference API key is available in the `ANTHROPIC_API_KEY` environment variable

## Task

Set up and run an LLM-as-a-Judge evaluation via the `nemo` CLI.

1. **Create a secret** named `nvidia-api-key` in the `eval-judge-workspace` workspace using: `nemo secrets create nvidia-api-key --workspace eval-judge-workspace --data "$ANTHROPIC_API_KEY" --description "NVIDIA inference API key"`

2. **Prepare and upload a dataset** - Create a JSONL file with `input` and `output` fields. Include at least one poor-quality response:
   ```jsonl
   {"input": "What is the capital of France?", "output": "The capital of France is Paris, which is also the largest city in France."}
   {"input": "How does photosynthesis work?", "output": "I don't know."}
   {"input": "What is 2+2?", "output": "2+2 equals 4."}
   ```
   Upload as fileset named `judge-eval-dataset` in `eval-judge-workspace`.

3. **Create an LLM-as-a-Judge metric** named `quality-judge` in `eval-judge-workspace`. Save this JSON to a file and use `nemo evaluation metrics create quality-judge --workspace eval-judge-workspace --input-file <path>`:
   ```json
   {
     "type": "llm-judge",
     "name": "quality-judge",
     "model": {
       "url": "https://inference-api.nvidia.com/v1",
       "name": "nvidia/openai/gpt-oss-20b",
       "format": "openai",
       "api_key_secret": "nvidia-api-key"
     },
     "scores": [
       {
         "name": "relevance",
         "description": "Relevance and quality score from 1 to 5",
         "minimum": 1,
         "maximum": 5,
         "parser": {
           "type": "json",
           "json_path": "relevance"
         }
       }
     ],
     "prompt_template": {
       "messages": [
         {"role": "system", "content": "You are an expert evaluator. Rate the response on relevance and quality from 1-5. Return your score as JSON: {\"relevance\": <score>}"},
         {"role": "user", "content": "Question: {{item.input}}\nResponse: {{item.output}}"}
       ]
     }
   }
   ```

4. **Run a synchronous evaluation** using `quality-judge` against a small inline dataset (2-3 rows). The sync eval returns immediate per-row scores.

5. **Examine the scored results** - Verify each row got a relevance score in 1-5 range and that good responses scored higher than poor ones.

6. **Create a metric evaluation job** referencing `quality-judge` and `judge-eval-dataset`. It may stay in "created" status - that is expected.

7. **Check the job status** to confirm it was created.
