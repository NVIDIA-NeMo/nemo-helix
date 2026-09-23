<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Error Mapping Framework

## Overview

The Error Mapping Framework provides a **rule-based exception mapping system** that converts low-level library exceptions into user-friendly, domain-specific errors.

---

## Problem Statement

### The Challenge

When building services that integrate with external libraries, we encounter a fundamental problem: **low-level libraries throw cryptic exceptions that are meaningless to end users**.

```python
# What framework code throws:
RuntimeError: CUDA error: out of memory
Tried to allocate 2.00 GiB (GPU 0; 15.78 GiB total capacity)

# What the user sees:
{"error": "RuntimeError: CUDA error: out of memory"}
```

This is problematic because:

1. **Users can't understand the error** - "CUDA error" means nothing to someone trying to fine-tune a model
2. **Users can't take action** - The error doesn't tell them how to fix it (reduce batch size, use fewer GPUs, etc.)
3. **Errors leak implementation details** - Internal library names, stack traces, and memory addresses are exposed
4. **No consistent error taxonomy** - Different libraries have different error hierarchies

### The Solution

This framework provides a **rule-based exception mapping system** that:

1. **Catches low-level exceptions** from libraries
2. **Matches them against configurable rules** (regex, keywords, type, etc.)
3. **Converts them to domain-specific exceptions** with user-friendly messages
4. **Allows rules to be updated without code changes** via YAML/JSON configuration

### Before and After

```
BEFORE (raw library exception):
  RuntimeError: CUDA error: out of memory
  Tried to allocate 2.00 GiB (GPU 0; 15.78 GiB total capacity)

AFTER (converted domain exception):
  CudaError:
    message: "GPU memory exhausted. Try reducing batch_size or sequence_length."
    detail: "RuntimeError: CUDA error: out of memory..."
    user_message: "Training failed due to insufficient GPU memory."
```

---

## Requirements

### Functional Requirements

- Match exceptions by exact message string
- Match exceptions by regex pattern
- Match exceptions by substring (contains)
- Match exceptions by exception type (isinstance)
- Match exceptions by exception class name (string)
- Match exceptions by multiple keywords (all/any)
- Combine matchers with AND/OR/NOT logic
- Match on chained exception (`__cause__`)
- Match on exception attributes (e.g., `errno`)
- Load rules from YAML configuration
- Load rules from JSON configuration
- Support custom handler functions for complex conversion logic
- Rules evaluated in order; first match wins
- Fallback to default handler when no rules match
- Re-raise original exception when no rules match (optional)

### Non-Functional Requirements

- Generic and reusable across all NeMo Helix services
- No external dependencies except PyYAML (optional)
- Type-safe with full type hints
- Rules can be updated without code changes
- Secure - no `eval()` or arbitrary code execution from config
- Fast - minimal overhead for exception matching
- Well-documented with examples
- Testable - easy to unit test rules

---

## Goals and Non-Goals

### Goals

1. **Provide a generic exception mapping abstraction** that any service within nhx can use
2. **Support declarative rules** in YAML/JSON for easy updates
3. **Support programmatic rules** for simple cases or dynamic logic
4. **Handle complex matching scenarios** (regex, keywords, types, composition)
5. **Separate framework from service-specific rules** - this package has no domain knowledge
6. **Be secure** - no code execution from configuration files

### Non-Goals

1. **Define domain-specific exceptions** - each service defines its own
2. **Provide pre-built rules** - each service maintains its own rules
3. **Handle exception logging** - services handle their own logging
4. **Integrate with FastAPI** - services wire up exception handlers themselves
5. **Support dynamic rule reloading** - reload converter on config change is service responsibility
6. **Provide metrics/observability** - services add their own instrumentation

---

## Architecture

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         THIS FRAMEWORK (nhx.common.errors)                     │
│                                                                                │
│  ┌─────────────────┐  ┌────────────────────┐  ┌─────────────────────────────┐  │
│  │    Matchers     │  │     Converter      │  │         Loader              │  │
│  │                 │  │                    │  │                             │  │
│  │  ExactMatcher   │  │ ExceptionConverter │  │ RulesLoader                 │  │
│  │  RegexMatcher   │  │                    │  │                             │  │
│  │  ContainsMatcher│  │ - add_rule()       │  │ - from_yaml()               │  │
│  │  TypeMatcher    │  │ - convert()        │  │ - from_json()               │  │
│  │  OrMatcher      │  │ - convert_or_*     │  │ - from_dict()               │  │
│  │  AndMatcher     │  │                    │  │                             │  │
│  │  CauseMatcher   │  │                    │  │                             │  │
│  │  ...            │  │                    │  │                             │  │
│  └─────────────────┘  └────────────────────┘  └─────────────────────────────┘  │
│                                                                                │
│  Types: Handler, ExceptionRegistry, HandlerRegistry                            │
└────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Uses
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SERVICE RESPONSIBILITY (e.g. Customizer)                │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────────────────┐    │
│  │ Exception       │  │ Rules YAML      │  │ Custom Handlers           │    │
│  │ Classes         │  │                 │  │ (optional)                │    │
│  │                 │  │ error_rules.yaml│  │                           │    │
│  │ ModelLoadError  │  │                 │  │ def cuda_oom_handler(e):  │    │
│  │ CudaError       │  │ - exact: "..."  │  │     # parse memory size   │    │
│  │ TimeoutError    │  │   exception: X  │  │     return CudaError(...) │    │
│  │ DatasetError    │  │ - regex: "..."  │  │                           │    │
│  │                 │  │   handler: Y    │  │                           │    │
│  └─────────────────┘  └─────────────────┘  └───────────────────────────┘    │
│                                                                             │
│  Registries: EXCEPTION_REGISTRY, HANDLER_REGISTRY                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Original   │     │   Matcher    │     │   Handler    │     │  Converted   │
│  Exception   │ ──▶ │  Evaluation  │ ──▶ │  Invocation  │ ──▶ │  Exception   │
│              │     │              │     │              │     │              │
│ RuntimeError │     │ Rules 1..N   │     │ Creates new  │     │ CudaError    │
│ "CUDA OOM"   │     │ First match  │     │ exception    │     │ "GPU OOM"    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## Component Details

### 1. Types (`types.py`)

Defines type aliases used throughout the framework:

```python
# Generic type variable for exception subclasses
TException = TypeVar("TException", bound=Exception)

# A handler function that creates a new exception from an original
Handler = Callable[[Exception], TException]

# Maps exception class names (strings in YAML) to actual Python classes
ExceptionRegistry = dict[str, type[Exception]]

# Maps handler names (strings in YAML) to handler callables
HandlerRegistry = dict[str, Handler]

# Default handler used by RulesLoader for creating exceptions
# Signature: (exception_class, original_exception, error_details) -> Exception
DefaultExceptionHandler = Callable[[type[Exception], Exception, str | None], Exception]
```

**Why Handler?**

The handler pattern allows flexible exception creation:
- Simple case: `lambda e: MyError(str(e))`
- Complex case: Parse error message, extract details, create rich exception

### 2. Matchers (`matchers.py`)

Abstract base class and concrete implementations for matching exceptions.

**Design Principles**:
- Each matcher answers ONE question: "Does this exception match my criteria?"
- Matchers are stateless (except compiled regex)
- Matchers are composable (AND, OR, NOT)

**Matcher Hierarchy**:

```
ExceptionMatcher (ABC)
├── Basic Matchers
│   ├── ExactMatcher          - str(exception) == pattern
│   ├── RegexMatcher          - re.search(pattern, str(exception))
│   ├── ContainsMatcher       - pattern in str(exception)
│   ├── StartsWithMatcher     - str(exception).startswith(pattern)
│   └── EndsWithMatcher       - str(exception).endswith(pattern)
├── Type Matchers
│   ├── ExceptionTypeMatcher  - isinstance(exception, type)
│   └── ExceptionTypeNameMatcher - type(exception).__name__ == name
├── Keyword Matchers
│   ├── AllKeywordsMatcher    - all(kw in msg for kw in keywords)
│   └── AnyKeywordMatcher     - any(kw in msg for kw in keywords)
├── Composite Matchers
│   ├── CompositeMatcher      - all(m.matches(e) for m in matchers)  [AND]
│   ├── OrMatcher             - any(m.matches(e) for m in matchers)  [OR]
│   └── NotMatcher            - not inner.matches(e)
└── Special Matchers
    ├── CauseMatcher          - matches exception.__cause__
    └── AttributeMatcher      - getattr(exception, attr) == value
```

### 3. Converter (`converter.py`)

The core rule engine that holds rules and converts exceptions.

**Key Behaviors**:
- Rules are evaluated in order
- First matching rule wins
- Handler is invoked with the original exception
- Returns converted exception or None/raises

**Methods**:

**Constructor**:
```python
ExceptionConverter(
    rules: list[tuple[ExceptionMatcher, Handler]] | None = None,
    default_handler: DefaultExceptionHandler | None = None,  # Set once, reuse in raise_converted_or_default
)
```

| Method | Behavior |
|--------|----------|
| `convert(exception)` | Returns converted exception or `None` |
| `raise_converted_or_original(exception)` | Always raises: converted (with `__cause__`) or original |
| `raise_converted_or_default(exception, default=None)` | Always raises: converted or default (both with `__cause__`). Uses constructor's `default_handler` if not overridden. |

### 4. Loader (`loader.py`)

Loads rules from YAML/JSON into an ExceptionConverter.

**Parameters**:

| Parameter | Required | Description |
|-----------|----------|-------------|
| `exception_registry` | **Yes** | Maps exception class names to Python classes |
| `handler_registry` | No | Maps handler names to handler callables |
| `default_handler` | No | Handler for creating exceptions. Used for both matched rules without a custom handler, and the fallback when no rule matches. Signature: `(exc_class, original, error_details) -> Exception` |
| `fallback_exception` | No | Exception class to use when no rule matches. If provided, the converter's `default_handler` is pre-configured to create this exception type, so `raise_converted_or_default()` can be called without arguments. |
| `fallback_modules` | No | Modules to search for exception types (default: `[builtins, subprocess]`) |

**Why Registries?**

YAML/JSON can only contain strings, not Python objects. Registries bridge the gap:

```yaml
# In YAML - just a string
exception: ModelLoadError

# Python loads as
{"exception": "ModelLoadError"}  # String, not a class!

# Registry resolves it
EXCEPTION_REGISTRY["ModelLoadError"]  # → ModelLoadError class
```

**Why not use `eval()` or `globals()`?**
- `eval()`: Security risk (arbitrary code execution)
- `globals()`: Fragile (only works in same module)
- Registry: Safe, explicit, controlled

---

## API Reference

### Code-based API (ExceptionConverter)

Build rules in Python code:

```python
from nhx.common.errors import ExceptionConverter, AllKeywordsMatcher

converter = ExceptionConverter()

# Convenience methods
converter.add_exact("Connection refused", lambda e: NetworkError(str(e)))
converter.add_regex(r"Timeout after \d+ seconds", lambda e: TimeoutError(str(e)))
converter.add_contains("CUDA", lambda e: CudaError(str(e)))
converter.add_type(ValueError, lambda e: ValidationError(str(e)))

# Custom matcher
converter.add_rule(
    AllKeywordsMatcher(["CUDA", "memory"]),
    lambda e: CudaError("GPU out of memory", detail=str(e))
)

# Convert exceptions
result = converter.convert(exception)              # None if no match
converter.raise_converted_or_original(exception)   # Always raises with __cause__ set
converter.raise_converted_or_default(exception, default_handler)  # Always raises with __cause__
```

### Config-based API (RulesLoader)

Load rules from configuration:

```python
from nhx.common.errors import RulesLoader

# Define registries
EXCEPTION_REGISTRY = {
    "NetworkError": NetworkError,
    "CudaError": CudaError,
}

HANDLER_REGISTRY = {
    "cuda_oom_handler": my_cuda_oom_handler,
}

# Load from YAML
converter = RulesLoader.from_yaml(
    "error_rules.yaml",
    exception_registry=EXCEPTION_REGISTRY,
    handler_registry=HANDLER_REGISTRY,  # Optional
    fallback_exception=InternalError,   # Optional - exception class for "no match" fallback
)

# Also supports JSON and dict
converter = RulesLoader.from_json("rules.json", EXCEPTION_REGISTRY)
converter = RulesLoader.from_dict(config_dict, EXCEPTION_REGISTRY)

# With fallback_exception, raise_converted_or_default() works without passing a handler:
try:
    some_library_call()
except Exception as e:
    converter.raise_converted_or_default(e)  # Raises matched exception or InternalError
```

**Custom Fallback Modules**

By default, exception types in rules (e.g., `type: ValueError`) are resolved from `builtins` and `subprocess`. You can specify additional modules to search:

```python
import ssl
import torch.cuda

converter = RulesLoader.from_yaml(
    "error_rules.yaml",
    exception_registry=EXCEPTION_REGISTRY,
    fallback_modules=[builtins, ssl, torch.cuda],  # Custom modules
)
```

This allows rules to reference exceptions like `SSLError` or `OutOfMemoryError` without adding them to the registry.

**Default Handler and Fallback Exception**

The `default_handler` and `fallback_exception` parameters work together:

| Parameter | Purpose |
|-----------|---------|
| `default_handler` | **How** to create exceptions (signature: `(exc_class, original, error_details) -> Exception`) |
| `fallback_exception` | **What** exception class to use when no rule matches |

The same `default_handler` is used in two scenarios:

1. **Rule matches, no custom handler**: `default_handler(matched_exc_class, original, error_details)`
2. **No rule matches**: `default_handler(fallback_exception, original, None)`

```python
# Example: both scenarios use the same handler
converter = RulesLoader.from_yaml(
    "rules.yaml",
    exception_registry=EXCEPTION_REGISTRY,
    fallback_exception=InternalError,  # Used when no rule matches
    # default_handler is optional - uses built-in handler if not specified
)

try:
    risky_operation()
except Exception as e:
    # If a rule matches → converts to matched exception class
    # If no rule matches → converts to InternalError
    converter.raise_converted_or_default(e)
```

---

## Configuration Format

### Rule Structure

Each rule has: **ONE matcher field** + **exception fields**

```yaml
- <matcher_field>: <value>       # MATCHER: when to trigger (pick ONE)
  exception: MyException         # EXCEPTION: class to create (REQUIRED)
  error_details: "User message"  # EXCEPTION: optional message for default handler
  handler: my_handler            # EXCEPTION: optional custom handler
```

### Exception Fields

| Field | Required | Description |
|-------|----------|-------------|
| `exception` | **Yes** | Exception class name from `exception_registry` |
| `error_details` | No | User-friendly message for default handler |
| `handler` | No | Custom handler name from `handler_registry` |

> **Note**: `error_details` and `handler` are **mutually exclusive**. Specifying both will raise a `ValueError`. Use `error_details` for simple message overrides, or `handler` for custom conversion logic. Below are some examples:

```yaml
# Simple message override (uses default handler)
- contains: "CUDA"
  exception: CudaError
  error_details: "GPU out of memory. Reduce batch size."

# Custom handler that can have a user friendly message defined
- contains: "CUDA"
  exception: CudaError
  handler: cuda_oom_handler

# This is invalid and will raise ValueError
- contains: "CUDA"
  exception: CudaError
  handler: cuda_oom_handler
  error_details: "This is not allowed!"
```

### Matcher Fields

**Message Matching**:
| Field | Type | Description |
|-------|------|-------------|
| `exact` | `str` | Message equals string exactly |
| `regex` | `str` | Message matches regex pattern |
| `contains` | `str` | Message contains substring |
| `starts_with` | `str` | Message starts with prefix |
| `ends_with` | `str` | Message ends with suffix |
| `all_keywords` | `[str]` | Message contains ALL keywords |
| `any_keywords` | `[str]` | Message contains ANY keyword |

**Type Matching**:
| Field | Type | Description |
|-------|------|-------------|
| `type` | `str` | Exception is instance of type |
| `type_name` | `str` | Exception class name equals string |

**Composite**:
| Field | Type | Description |
|-------|------|-------------|
| `and` | `[matcher]` | ALL sub-matchers must match |
| `or` | `[matcher]` | ANY sub-matcher must match |
| `not` | `matcher` | Sub-matcher must NOT match |

**Special**:
| Field | Type | Description |
|-------|------|-------------|
| `cause` | `matcher` | Match on `__cause__` |
| `attribute` | `{name, value}` | Match attribute value |

### Complete Example

```yaml
# error_rules.yaml
rules:
  # Exact match
  - exact: "Connection refused"
    exception: NetworkError
    error_details: "Could not connect to server"

  # Regex with custom handler
  - regex: "^Timeout after (\\d+) seconds$"
    exception: TimeoutError
    handler: timeout_with_duration

  # Keyword matching
  - all_keywords: ["CUDA", "out of memory"]
    exception: CudaError
    error_details: "GPU OOM. Reduce batch_size."

  # Type matching
  - type_name: OutOfMemoryError
    exception: CudaError

  # Composite: RuntimeError containing "distributed"
  - and:
      - type_name: RuntimeError
      - contains: "distributed"
    exception: DistributedError

  # Match chained exception
  - cause:
      type: TimeoutError
    exception: TrainingTimeoutError
```

---

## Best Practices

### 1. Order Rules from Specific to General

```yaml
rules:
  # Specific first
  - exact: "CUDA out of memory on device 0"
    exception: CudaOOMDevice0Error

  # Then general
  - contains: "CUDA out of memory"
    exception: CudaError

  # Catch-all last (if using)
  - type_name: RuntimeError
    exception: InternalError
```

### 2. Use Appropriate Matcher Types

| Scenario | Recommended Matcher |
|----------|---------------------|
| Known exact message | `exact` |
| Pattern with variables | `regex` |
| Known substring | `contains` |
| Multiple indicators | `all_keywords` or `any_keywords` |
| Exception hierarchy | `type` |
| Third-party exceptions | `type_name` (no import needed) |

### 3. Design Exception Classes for API Responses

```python
class MyServiceError(Exception):
    """Base exception with API-friendly fields."""
    
    # For HTTP response
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"
    
    # For user display
    user_message: str = "An error occurred."
    
    def __init__(self, message: str, detail: str = None):
        self.message = message  # Developer message
        self.detail = detail    # Original error
        super().__init__(message)

class CudaError(MyServiceError):
    status_code = 400
    error_code = "GPU_ERROR"
    user_message = "Training failed due to GPU issues."
```

### 4. Use Custom Handlers for Complex Logic

```python
def cuda_oom_handler(original: Exception) -> CudaError:
    """Parse CUDA OOM details and provide specific guidance."""
    msg = str(original)
    
    # Extract memory size
    match = re.search(r"(\d+\.?\d*)\s*(GB|MB)", msg)
    if match:
        size, unit = match.groups()
        return CudaError(
            message=f"GPU requires {size} {unit} but ran out. Reduce batch_size.",
            detail=msg,
        )
    
    # Extract GPU device
    match = re.search(r"device (\d+)", msg)
    if match:
        device = match.group(1)
        return CudaError(
            message=f"GPU {device} ran out of memory.",
            detail=msg,
        )
    
    return CudaError(message="GPU out of memory.", detail=msg)
```

---

## Limitations

1. **First-match-wins only**: No priority/weight system. Order rules carefully.

2. **No dynamic rule reloading**: Converter is built once. Service must recreate for new rules.

3. **String-based matching**: Cannot match on exception fields other than message (except `AttributeMatcher`).

4. **Single exception output**: Cannot map one exception to multiple exceptions.

5. **No async support needed**: Exception handling is synchronous.

---

## Appendix A: Generic Service Integration Example

### 1. Exception Classes

```python
# myservice/exceptions.py

class MyServiceError(Exception):
    """Base exception for MyService."""
    status_code: int = 500
    error_code: str = "SERVICE_ERROR"
    user_message: str = "An error occurred."

    def __init__(self, message: str, detail: str = None):
        self.message = message
        self.detail = detail
        super().__init__(message)


class NetworkError(MyServiceError):
    status_code = 503
    error_code = "NETWORK_ERROR"
    user_message = "Network connectivity issue."


class InternalError(MyServiceError):
    status_code = 500
    error_code = "INTERNAL_ERROR"
    user_message = "An unexpected error occurred."
```

### 2. Error Rules YAML

```yaml
# myservice/error_rules.yaml
rules:
  - contains: "Connection refused"
    exception: NetworkError
    error_details: "Could not connect to remote service."

  - contains: "timeout"
    exception: NetworkError
    error_details: "Connection timed out."
```

### 3. Error Handler

```python
# myservice/error_handler.py

from nhx.common.errors import RulesLoader
from .exceptions import NetworkError, InternalError

EXCEPTION_REGISTRY = {
    "NetworkError": NetworkError,
    "InternalError": InternalError,
}

converter = RulesLoader.from_yaml(
    "error_rules.yaml",
    exception_registry=EXCEPTION_REGISTRY,
    fallback_exception=InternalError,
)

def handle_error(exception: Exception) -> None:
    """Convert and re-raise any exception."""
    converter.raise_converted_or_default(exception)
```

### 4. Usage

```python
try:
    external_library_call()
except Exception as e:
    handle_error(e)  # Raises NetworkError or InternalError
```

---

## Appendix B: Customizer Service Integration Example

The Customizer service presents a unique challenge: **training runs in a subprocess**, and errors are lost at the subprocess boundary.

### The Subprocess Problem

```
┌─────────────────────────────────────────────────────────────────┐
│  Main Process (train.py / train_automodel.py / train_rl.py)     │
│                                                                 │
│  training_subprocess = subprocess.Popen(command,                │
│                                        stderr=subprocess.STDOUT)│
│  training_subprocess.wait(timeout=timeout)                      │
│                                                                 │
│  if returncode != 0:                                            │
│      err = Exception("Training subprocess returned with         │
│                       error code: {returncode}")  ← GENERIC!    │
│      callback.report_exception(err, ...)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ subprocess.Popen()
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Subprocess (automodel_finetune.py / run_grpo_penguin.py)       │
│                                                                 │
│  def main():                                                    │
│      grpo_train(...)  # ← Framework errors here                 │
│                                                                 │
│  # NO try/except - errors just crash and print to stderr        │
│  # stderr → stdout (buried in container logs)                   │
└─────────────────────────────────────────────────────────────────┘
```

**What users see today:**
```json
{"status": "FAILED", "error": "Training subprocess returned with error code: 1"}
```

**What's actually in the container logs (users can't access):**
```
RuntimeError: CUDA error: out of memory
Traceback (most recent call last):
  File "/app/nemo_rl/algorithms/grpo.py", line 583
  ...
```

### Why This Happens

1. **No Error Capture in Subprocess**: The actual training scripts (`automodel_finetune.py`, `run_grpo_penguin.py`) have no `try/except` to catch framework errors

2. **Callback Only Works in Main Process**: `AutomodelCustomizerCallback.report_exception()` is only called from the main process after the subprocess dies

3. **PyTorch Lightning Limitation**: `NeMoCustomizerCallback.on_exception()` only works when using PyTorch Lightning trainer (NeMo SFT), not for Automodel or NeMo-RL

4. **stderr Goes to Void**: `stderr=subprocess.STDOUT` merges stderr into stdout, but it's not captured or parsed

### The Solution: Integrate Inside Subprocess

To solve this, the error mapping must be integrated **inside** the subprocess:

```python
# In automodel_finetune.py or run_grpo_penguin.py (inside subprocess)
from nhx.common.errors import RulesLoader
from customizer_training.clients.automodel_callback import AutomodelCustomizerCallback
from customizer.exceptions import InternalError

def main():
    callback = AutomodelCustomizerCallback()
    converter = RulesLoader.from_yaml(
        "error_rules.yaml",
        exception_registry=EXCEPTION_REGISTRY,
        fallback_exception=InternalError,
    )
    
    try:
        recipe = CustomizerTrainFinetuneRecipe(cfg)
        recipe.setup()
        recipe.run_train_validation_loop()
    except Exception as e:
        # Convert and report the error to the API
        try:
            converter.raise_converted_or_default(e)
        except Exception as converted:
            callback.report_exception(converted, "training error", detail=str(e))
            raise  # Re-raise converted exception (will set non-zero exit code)
```

This ensures:
1. Framework errors are caught before subprocess exits
2. Errors are converted to user-friendly messages
3. Errors are reported to the API via callback
4. Users see actionable error messages instead of generic "error code: 1"

### After Integration

```json
{
  "status": "FAILED", 
  "error": {
    "code": "CUDA_ERROR",
    "message": "GPU memory exhausted. Try reducing batch_size or max_seq_length.",
    "detail": "RuntimeError: CUDA error: out of memory. Tried to allocate 2.00 GiB..."
  }
}
```
