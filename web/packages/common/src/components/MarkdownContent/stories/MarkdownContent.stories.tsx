// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MarkdownContent } from '@nemo/common/src/components/MarkdownContent';
import type { Meta, StoryObj } from '@storybook/react';

const meta = {
  component: MarkdownContent,
  title: 'Studio Common/MarkdownContent',
  // Mirror the surface MarkdownContent renders in today (`ModelReadme`):
  // a padded raised card. Keeps stories visually representative of production.
  decorators: [
    (Story) => (
      <div className="rounded-lg border border-base bg-surface-raised p-density-xl">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof MarkdownContent>;

export default meta;
type Story = StoryObj<typeof meta>;

const KITCHEN_SINK = `# Llama 3.1 8B Instruct

Llama 3.1 is Meta's collection of multilingual large language models. The 8B
**instruction-tuned** variant is optimized for *dialogue* and tool use, with a
\`128k\` context window.

## Quickstart

Install the SDK:

\`\`\`bash
pip install llama-cpp-python
\`\`\`

Then load the model:

\`\`\`python
from llama_cpp import Llama

llm = Llama(model_path="./llama-3.1-8b-instruct.gguf")
print(llm("Hello, world!"))
\`\`\`

## Capabilities

- Multilingual dialogue across 8 languages
- Tool calling
- Long-context summarization (up to 128k tokens)
  - Document QA
  - Codebase comprehension
- Function calling

### Recommended hyperparameters

| Parameter | Default | Range |
| --- | --- | --- |
| Temperature | 0.7 | 0.0 – 1.5 |
| Top-p | 0.9 | 0.0 – 1.0 |
| Max tokens | 512 | 1 – 8192 |

## Notes

> [!NOTE]
> Llama 3.1 weights are released under Meta's [community license](https://llama.meta.com/llama3_1/license).

> [!TIP]
> For best latency, deploy with vLLM and enable continuous batching.

> [!WARNING]
> The base model is not safety-tuned. Pair it with a guardrails layer in production.

---

See the [model card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) for benchmarks.
`;

/** A representative README exercising headings, prose, code, lists, tables, callouts, and links. */
export const KitchenSink: Story = {
  args: { content: KITCHEN_SINK },
};

/** Every heading level (h1–h6) in order to show the title-kind ramp. */
export const Headings: Story = {
  args: {
    content: `# Heading 1 — title/2xl
## Heading 2 — title/xl
### Heading 3 — title/lg
#### Heading 4 — title/md
##### Heading 5 — title/sm
###### Heading 6 — title/xs

A paragraph in body/regular/md follows the headings to show their relative scale.
`,
  },
};

/** Paragraph-level prose: bold, italic, inline code, and links. */
export const InlineFormatting: Story = {
  args: {
    content: `Inline code like \`useState\` should monospace via Foundations' inline
\`CodeSnippet\`. **Bold text** and *italic text* render via the platform's
default text styling.

Links open in a new tab: see the [react-markdown docs](https://github.com/remarkjs/react-markdown)
or [the GFM spec](https://github.github.com/gfm/) for details.
`,
  },
};

/** Ordered, unordered, nested, and GFM task lists. */
export const Lists: Story = {
  args: {
    content: `### Unordered with nesting

- Item one
- Item two
  - Nested item A
  - Nested item B
    - Even deeper
- Item three

### Ordered

1. First step
2. Second step
3. Third step

### Task list (GFM)

- [x] Land MarkdownContent component
- [x] Wire it into ModelReadme
- [ ] Add Storybook stories
- [ ] Write unit tests
`,
  },
};

/**
 * Fenced code blocks across several scenarios — supported language, unsupported
 * language (falls back to `'markdown'`), no language, and inline code.
 */
export const CodeBlocks: Story = {
  args: {
    content: `### TypeScript (supported — Shiki highlights)

\`\`\`typescript
interface Greeting {
  hello: string;
}

const greet = ({ hello }: Greeting): string => \`Hello, \${hello}!\`;

console.log(greet({ hello: 'world' }));
\`\`\`

### Python (supported — Shiki highlights)

\`\`\`python
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
\`\`\`

### Dockerfile (unsupported — falls back to \`markdown\` grammar)

\`\`\`dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
\`\`\`

### No language (falls back to \`markdown\` grammar)

\`\`\`
just some
plain text
in a fenced block
\`\`\`

### Inline code

A sentence with \`inline code\` — Foundations renders this with the inline
\`CodeSnippet\` variant.
`,
  },
};

/** GFM table with header row, body rows, and varied content widths. */
export const Tables: Story = {
  args: {
    content: `| Model | Parameters | Context | Notes |
| --- | --- | --- | --- |
| llama-3.1-8b-instruct | 8B | 128k | General dialogue |
| nemotron-nano-llama-3.1-8b | 8B | 4k | Compact, instruction-tuned |
| phi-4 | 14B | 16k | Strong reasoning, small footprint |
`,
  },
};

/** All five GitHub-style alert kinds. */
export const Callouts: Story = {
  args: {
    content: `> [!NOTE]
> Notes call out neutral information that complements the surrounding content.

> [!TIP]
> Tips highlight a recommended approach.

> [!IMPORTANT]
> Important alerts flag something the reader must not miss.

> [!WARNING]
> Warnings call out behavior that may have unexpected consequences.

> [!CAUTION]
> Cautions call out behavior that may cause harm or data loss.
`,
  },
};

/** A plain blockquote with no callout marker — falls through to the default styling. */
export const PlainBlockquote: Story = {
  args: {
    content: `> "The best way to predict the future is to invent it."
> — Alan Kay

Followed by a normal paragraph.
`,
  },
};

/** A short, single-paragraph render to verify minimal-content layout. */
export const Minimal: Story = {
  args: {
    content: `A single line of markdown content with **emphasis**.`,
  },
};
