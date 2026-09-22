---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: langchain-framework
description: "Guidance for building LLM-powered agents with LangChain, LangGraph, or Deep Agents. Use when building agents with any of these frameworks, implementing tool-calling agents, stateful graph workflows, or multi-step orchestration. Read this skill before writing any LangChain/LangGraph agent code."
compatibility: Python ≥ 3.10, uv for install, API keys in .env. langchain >= 1.0,<2.0; langgraph >= 1.0,<2.0; langsmith >= 0.3.0; deepagents latest.
allowed-tools: Read
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Building Agents with LangChain / LangGraph

> NVIDIA-authored guidance summarizing the public LangChain, LangGraph, and Deep Agents
> documentation (<https://docs.langchain.com>), which is published by LangChain, Inc. under the
> MIT license. API names and short usage examples are drawn from those docs; the framework
> selection guidance, comparisons, and structure are original.

LangChain, LangGraph, and Deep Agents are **layered** frameworks — each builds on the one below:

```
┌─────────────────────────────────────────┐
│              Deep Agents                │  ← batteries included: planning, memory, files
├─────────────────────────────────────────┤
│               LangGraph                 │  ← orchestration: graphs, loops, state
├─────────────────────────────────────────┤
│               LangChain                 │  ← foundation: models, tools, chains
└─────────────────────────────────────────┘
```

## Pick Your Framework

| Question | Yes → | No → |
|----------|-------|------|
| Need sub-task planning, file management, persistent memory, or on-demand skills? | **Deep Agents** | ↓ |
| Need loops, branching, human-in-the-loop, or custom state? | **LangGraph** | ↓ |
| Single-purpose agent with fixed tools? (+ middleware for HITL, retry, PII, etc.) | **LangChain** (`create_agent`) | ↓ |
| Pure model call or chain with no agent loop? | **LangChain** (LCEL) | — |

## Installation

```bash
uv add langchain langchain-core langsmith

# Orchestration — only add if you use langgraph.* imports directly or Deep Agents:
uv add langgraph          # StateGraph, checkpointers, etc.
uv add deepagents         # Deep Agents (includes LangGraph)

# Model provider — pick yours:
uv add langchain-anthropic   # Claude (direct) — installed in this environment
uv add langchain-openai      # GPT or any OpenAI-compatible endpoint — NOT pre-installed; add explicitly
```

> **Note:** `langchain-openai` is **not installed by default** in this environment. Code examples below that use `ChatOpenAI` require `uv add langchain-openai` first. All code examples work with `langchain-anthropic` out of the box.

Set your API key in `.env`:
```bash
ANTHROPIC_API_KEY=<your-key>   # if using langchain-anthropic directly
# or, for any OpenAI-compatible endpoint (e.g. a custom inference server):
OPENAI_API_KEY=<your-key>
LANGSMITH_API_KEY=<your-key>   # optional, for tracing
```

## Hello World

### LangChain — LCEL chain (pure model call, no agent loop)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
import os

# Option A: Anthropic directly (langchain-anthropic installed by default)
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-sonnet-4-5")

# Option B: OpenAI-compatible endpoint — requires: uv add langchain-openai
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model="<model-name>",
    base_url="<your-endpoint-url>",
    api_key=os.environ["YOUR_API_KEY"],
)

# Option 1: plain text output
chain = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
]) | model | StrOutputParser()

print(chain.invoke({"input": "Say hello"}))

# Option 2: structured output
class Reply(BaseModel):
    message: str

structured_chain = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
]) | model.with_structured_output(Reply)

print(structured_chain.invoke({"input": "Say hello"}))
```

### LangChain — `create_agent` (single-purpose agent with fixed tools)

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from datetime import date

@tool
def get_current_date() -> str:
    """Get today's date."""
    return date.today().isoformat()

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",   # or a model instance (see above)
    tools=[get_current_date],
    system_prompt="You are a helpful assistant.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "What is today's date?"}]})
print(result["messages"][-1].content)
```

**Middleware** — `create_agent` supports a `middleware` list for cross-cutting concerns:

| Middleware | Purpose |
|---|---|
| `HumanInTheLoopMiddleware` | Pause for human approval before tool calls |
| `TodoListMiddleware` | Track sub-tasks with a built-in todo list |
| `ModelRetryMiddleware` | Retry on model errors |
| `ToolRetryMiddleware` | Retry on tool call errors |
| `ModelCallLimitMiddleware` | Cap total LLM calls per run |
| `SummarizationMiddleware` | Auto-compress context when it grows too long |
| `PIIMiddleware` | Strip PII from model inputs/outputs |

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, TodoListMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[...],
    middleware=[HumanInTheLoopMiddleware(), TodoListMiddleware()],
)
```

**Per-invocation typed config (`context_schema`)** — pass typed context (user IDs, flags) to nodes without global state:

```python
from langchain.agents import create_agent
from pydantic import BaseModel

class UserContext(BaseModel):
    user_id: str
    locale: str = "en"

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[...],
    context_schema=UserContext,
)

result = agent.invoke(
    {"messages": [...]},
    config={"configurable": {"context": UserContext(user_id="u123")}}
)
```

See `references/langchain-middleware.md` for full middleware reference.

### LangGraph — `StateGraph` (custom state, loops, branching)

```python
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from datetime import date
import os

# Requires: uv add langchain-openai  (or swap for ChatAnthropic)
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model="<model-name>",
    base_url="<your-endpoint-url>",
    api_key=os.environ["YOUR_API_KEY"],
)

@tool
def get_current_date() -> str:
    """Get today's date."""
    return date.today().isoformat()

class State(TypedDict):
    messages: Annotated[list, add_messages]

model_with_tools = model.bind_tools([get_current_date])

def call_model(state: State) -> State:
    return {"messages": [model_with_tools.invoke(state["messages"])]}

graph = StateGraph(State)
graph.add_node("agent", call_model)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)
agent = graph.compile()

result = agent.invoke({"messages": [{"role": "user", "content": "What is today's date?"}]})
print(result["messages"][-1].content)
```

### Deep Agents — `create_deep_agent` (planning, files, memory, subagents)

```python
from deepagents import create_deep_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from datetime import date

@tool
def get_current_date() -> str:
    """Get today's date."""
    return date.today().isoformat()

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[get_current_date],
    system_prompt="You are a helpful assistant.",
    checkpointer=MemorySaver(),  # required for deep agents
)

# thread_id is always required for deep agents
config = {"configurable": {"thread_id": "session-1"}}
result = agent.invoke({"messages": [{"role": "user", "content": "What is today's date?"}]}, config=config)
print(result["messages"][-1].content)
```

## Core Concepts

### Tools

Tools are Python functions decorated with `@tool`. The docstring is the LLM's description of when and how to call the tool. Pass tools to `create_agent`, `create_deep_agent`, or bind them to a LangGraph model with `bind_tools`.

```python
from langchain_core.tools import tool

@tool
def search_files(pattern: str) -> list[str]:
    """Search for files matching the given glob pattern."""
    import glob
    return glob.glob(pattern, recursive=True)

agent = create_agent(model="anthropic:claude-sonnet-4-5", tools=[search_files])
```

For stateful or resource-sharing tools, use class-based tools or `StructuredTool` for complex input schemas.

---

### System Prompt

Pass `system_prompt=` to `create_agent` or `create_deep_agent` to set the agent's role and constraints:

```python
agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[...],
    system_prompt="You are a precise code reviewer. Always cite line numbers.",
)
```

For LCEL chains, set the system message in `ChatPromptTemplate`:

```python
chain = ChatPromptTemplate.from_messages([
    ("system", "You are a precise code reviewer. Always cite line numbers."),
    ("human", "{input}"),
]) | model | StrOutputParser()
```

For LangGraph, inject a `SystemMessage` at the start of the state or inside a node.

---

### Agent Types

Choose based on workflow complexity:

| Question | Answer |
|---|---|
| Need planning, memory, file management, or on-demand skills? | `create_deep_agent` |
| Need custom state, loops, branching, or human-in-the-loop? | `StateGraph` (LangGraph) |
| Single-purpose agent with fixed tools? | `create_agent` |
| Pure model call or chain? | LCEL (`ChatPromptTemplate | model | parser`) |

```python
# create_agent — fixed tools, middleware support
agent = create_agent(model="anthropic:claude-sonnet-4-5", tools=[...])

# StateGraph — custom state and control flow
graph = StateGraph(State)
graph.add_node("agent", call_model)
agent = graph.compile()

# create_deep_agent — planning, memory, subagents, skills
agent = create_deep_agent(model="...", tools=[...], checkpointer=MemorySaver())
```

---

### Structured Output

When a method needs to return a specific format, use `.with_structured_output()` rather than describing the format in the system prompt. Structured output is enforced by the model API — it is more reliable than prompt instructions and gives downstream code a typed, predictable value.

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float
    topics: list[str]

structured_chain = ChatPromptTemplate.from_messages([
    ("system", "Analyze the text."),
    ("human", "{input}"),
]) | model.with_structured_output(Analysis)

result = structured_chain.invoke({"input": "Great product!"})
print(result.sentiment)  # validated Analysis instance
```

Bind the schema in `__init__` so every call on that model instance returns the typed output:

```python
class MyAgent:
    def __init__(self):
        self.llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct").with_structured_output(MySchema)
```

---

### Dynamic Prompts

Use template variables in `ChatPromptTemplate` to inject runtime values:

```python
chain = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Focus on {domain}."),
    ("human", "{input}"),
]) | model | StrOutputParser()

result = chain.invoke({"role": "code reviewer", "domain": "security", "input": "..."})
```

For `create_agent`, pass per-invocation typed config via `context_schema`:

```python
class RunContext(BaseModel):
    language: str = "Python"
    strict_mode: bool = False

agent = create_agent(model="...", tools=[...], context_schema=RunContext)
result = agent.invoke({...}, config={"configurable": {"context": RunContext(language="Go")}})
```

---

### Skills

Deep Agents supports skills — callable modules the agent can invoke on demand:

```python
from deepagents import create_deep_agent, Skill

def read_policy() -> str:
    return Path("policies/patch.md").read_text()

my_skill = Skill(name="patch-policy", description="Rules for writing correct patches.", fn=read_policy)

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[...],
    skills=[my_skill],
    checkpointer=MemorySaver(),
)
```

For `create_agent` or plain LangGraph, encode reusable knowledge as additional tools or inject it into the system prompt.

See `references/deep-agents-core.md` for the full skills API.

---

## Advanced Features

### Subagents

Expose a compiled agent as a `@tool` so the parent can delegate to it:

```python
from langchain_core.tools import tool

child_agent = create_agent(model="anthropic:claude-sonnet-4-5", tools=[...], system_prompt="...")

@tool
def delegate_to_child(request: str) -> str:
    """Delegate a focused sub-task to the specialized child agent."""
    result = child_agent.invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].content

parent_agent = create_agent(model="...", tools=[..., delegate_to_child])
```

For Deep Agents, see `references/deep-agents-orchestration.md` for the native subagent API.

---

### LLM Configuration

Use a model string shorthand or an explicit instance:

```python
# String shorthand
agent = create_agent(model="anthropic:claude-opus-4-7", tools=[...])

# Explicit instance — for custom endpoints
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="...", base_url="...", api_key=os.environ["KEY"])
agent = create_agent(model=model, tools=[...])
```

In LangGraph, each node can hold its own model instance for per-step model selection:

```python
fast_model = ChatOpenAI(model="gpt-5-mini", ...)
strong_model = ChatOpenAI(model="gpt-5", ...)

def quick_filter(state): return {"messages": [fast_model.invoke(state["messages"])]}
def deep_patch(state):   return {"messages": [strong_model.invoke(state["messages"])]}
```

---

### Middleware and Config

Add cross-cutting behavior via the `middleware` list:

```python
from langchain.agents.middleware import ModelRetryMiddleware, SummarizationMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[...],
    middleware=[ModelRetryMiddleware(max_retries=3), SummarizationMiddleware()],
)
```

Available middleware: `HumanInTheLoopMiddleware`, `TodoListMiddleware`, `ModelRetryMiddleware`, `ToolRetryMiddleware`, `ModelCallLimitMiddleware`, `SummarizationMiddleware`, `PIIMiddleware`.

Control LangGraph iteration depth with `recursion_limit`:

```python
agent = graph.compile(recursion_limit=50)
```

---

### Persistence

LangGraph checkpointers persist agent state across invocations. Pass `thread_id` to resume a session:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = MemorySaver()                                    # in-memory, lost on restart
checkpointer = SqliteSaver.from_conn_string("agent_state.db")  # survives restarts

agent = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "session-1"}}
result = agent.invoke({"messages": [...]}, config=config)
```

See `references/langgraph-persistence.md` for time-travel and Store APIs.

---

### MCP Integration

Use MCP (Model Context Protocol) servers as tool sources via `langchain-mcp-adapters`:

```bash
uv add langchain-mcp-adapters
```

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

async with MultiServerMCPClient({
    "my-server": {
        "url": "https://my-mcp-server.example.com/mcp",
        "transport": "streamable_http",
    }
}) as client:
    tools = await client.get_tools()
    agent = create_agent(model="anthropic:claude-sonnet-4-5", tools=tools)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": "..."}]})
```

For Managed Deep Agents hosted in LangSmith, MCP servers are registered and credential-managed by the platform. See `references/managed-deep-agents.md` for the hosted path.

---

### Tracing

Wire tracing via OpenInference + OpenTelemetry:

```python
from openinference.instrumentation.langchain import LangChainInstrumentor
LangChainInstrumentor().instrument()
```

For LangSmith, set environment variables before running:

```bash
LANGSMITH_API_KEY=<your-key>
LANGCHAIN_TRACING_V2=true
```

See `references/openinference-tracing.md` for file-based JSONL tracing.

---

## Reference Docs

Load the relevant reference when you need detailed guidance:

| Reference | When to load |
|-----------|-------------|
| [references/ecosystem-primer.md](references/ecosystem-primer.md) | Framework selection, env setup, docs navigation |
| [references/langchain-dependencies.md](references/langchain-dependencies.md) | Package versions, installation, environment setup |
| [references/langchain-fundamentals.md](references/langchain-fundamentals.md) | `create_agent()`, tool definition, structured output |
| [references/langchain-middleware.md](references/langchain-middleware.md) | Human-in-the-loop, approval workflows |
| [references/langchain-rag.md](references/langchain-rag.md) | RAG pipelines, vector stores, document loaders |
| [references/langgraph-fundamentals.md](references/langgraph-fundamentals.md) | StateGraph, nodes, edges, Send, Command |
| [references/langgraph-persistence.md](references/langgraph-persistence.md) | Checkpointers, thread_id, time travel, Store |
| [references/langgraph-human-in-the-loop.md](references/langgraph-human-in-the-loop.md) | `interrupt()`, `Command(resume=...)`, idempotency |
| [references/langgraph-cli.md](references/langgraph-cli.md) | langgraph CLI: dev, build, deploy, langgraph.json config |
| [references/deep-agents-core.md](references/deep-agents-core.md) | `create_deep_agent()`, built-in middleware, skills |
| [references/deep-agents-memory.md](references/deep-agents-memory.md) | State/Store/Filesystem/CompositeBackend |
| [references/deep-agents-orchestration.md](references/deep-agents-orchestration.md) | Subagents, TodoList, HITL |
| [references/managed-deep-agents.md](references/managed-deep-agents.md) | LangSmith hosted Deep Agents, deepagents-cli, useStream |
| [references/openinference-tracing.md](references/openinference-tracing.md) | Wiring file-based tracing via OpenInference + OpenTelemetry SDK |
