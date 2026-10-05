<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Example Test Template - Harbor Test

This is a template for creating new Harbor tests. Copy this folder and fill in the TODOs.

## Getting Started

1. Copy this folder to a new folder with a descriptive name:
   ```bash
   cp -r tests/agentic-use/example-test-template tests/agentic-use/my-new-test
   ```

2. Fill in all the TODOs in each file (see checklist below)

3. Set up your Anthropic API credentials:
   ```bash
   export ANTHROPIC_API_KEY='sk-your-api-key' # Get this from inference hub
   export ANTHROPIC_BASE_URL='https://inference-api.nvidia.com'
   ```

4. Run with Harbor:
   ```bash
   harbor run -p tests/agentic-use/my-new-test \
       --agent claude-code \
       --model anthropic/claude-sonnet-4-5
   ```

## Files to Customize

- **README.md** - Update this file to describe your specific test
- **instruction.md** - The task prompt given to Claude Code
- **task.toml** - Harbor configuration (timeouts, metadata, difficulty)
- **environment/Dockerfile** - Container setup (usually inherits from nhx-agentic-base:latest)
- **tests/test.sh** - Test runner script
- **tests/test_outputs.py** - Pytest verification of task completion

## TODO Checklist

Before running your test, ensure you've completed:

- [ ] Updated README.md with test description
- [ ] Defined the task in instruction.md
- [ ] Set metadata in task.toml (author, difficulty, category, tags)
- [ ] Configured timeouts in task.toml if defaults aren't suitable
- [ ] Written verification tests in tests/test_outputs.py

## Test Design Guidelines

1. **Clear Success Criteria** - Define exactly what constitutes success
2. **Atomic Tasks** - Test one capability at a time when possible
3. **Deterministic Verification** - Tests should not be flaky
4. **Appropriate Timeouts** - Set realistic timeouts for agent and verifier
5. **Good Instructions** - Be specific about what tools are available

## Difficulty Guidelines

- **easy** - Single operation, clear instructions (e.g., create one resource)
- **medium** - Multiple steps, some decision-making required
- **hard** - Complex multi-step tasks, error handling, or ambiguous requirements
