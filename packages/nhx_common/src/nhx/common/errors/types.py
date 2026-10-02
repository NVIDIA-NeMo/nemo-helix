# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Type aliases for the error mapping.
"""

from typing import Callable, TypeVar

TException = TypeVar("TException", bound=Exception)

# Type alias for exception handler functions
# A handler takes an original exception and returns a new (converted) exception
Handler = Callable[[Exception], TException]

# Type alias for exception registry
# Maps exception class names (strings from YAML/JSON) to actual Python classes
# This registry is provided by the calling service and is used to resolve the exception class name to the actual class.
ExceptionRegistry = dict[str, type[Exception]]

# Type alias for handler registry
# Maps handler names (strings from YAML/JSON) to actual handler callables
# This registry is provided by the calling service and is used to resolve the handler name to the actual handler callable.
# The Callable can be anything, a lambda, a function, a class method, etc. that takes an original exception and returns a new (converted) exception.
# The logic inside the handler is up to the calling service.
HandlerRegistry = dict[str, Handler]

# Type alias for the default handler used by RulesLoader
# This handler is used to construct exceptions for matched rules that don't have a custom handler.
# It's also used for the fallback when no rule matches (if fallback_exception is provided).
# Signature: (exception_class, original_exception, error_details) -> new_exception
DefaultExceptionHandler = Callable[[type[Exception], Exception, str | None], Exception]

# It controls what gets imported when the calling service uses the wildcard import.
__all__ = [
    "TException",
    "Handler",
    "ExceptionRegistry",
    "HandlerRegistry",
    "DefaultExceptionHandler",
]
