# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-based filter query parser using Lark (canonical home).

Query Syntax Examples:
    name:"llama"                                      - Exact match
    created>1620310503                                - Numeric comparison (timestamp)
    email~"amy"                                       - Substring/like match
    -currency:"jpy"                                   - Negation
    status:"active" AND amount>500                    - AND combination
    currency:"usd" OR currency:"eur"                  - OR combination
    metadata["key"]:"value"                           - Metadata field access
    data.nested.key:"value"                           - Nested field access
    url:null                                          - NULL check
    (name:"a" OR name:"b") AND status:"active"        - Nested grouping
    -(name:"llama" AND status:"draft")                - Negated group
    status IN ["active", "pending"]                   - IN list match
    status NOT IN ["deleted", "archived"]             - NOT IN list match

Date Comparisons:
    created>"2024-01-01"                              - After date (ISO format)
    created>="2024-01-01T10:00:00Z"                   - After datetime with timezone
    updated<"2024-12-31"                              - Before date
    created>"2024-01-01" AND created<"2024-12-31"     - Date range

Operators:
    :   - Exact match (case insensitive for tokens)
    ~   - Substring match (minimum 3 characters)
    >   - Greater than (numbers, timestamps, or ISO date strings)
    >=  - Greater than or equal
    <   - Less than
    <=  - Less than or equal
    IN  - Value in list
    NOT IN - Value not in list
    -   - Negation prefix
    ()  - Grouping for nested expressions

Logical Operators:
    AND - Combine clauses (all must match)
    OR  - Combine clauses (any must match)
    (space) - Implicit AND when no operator specified

Note: Use parentheses to nest AND/OR operations.
"""

import ast
from typing import Any, List, Union, cast

from lark import Lark, Transformer, v_args
from nemo_helix_plugin.api.filter import (
    ComparisonOperation,
    FilterOperation,
    FilterOperator,
    LogicalOperation,
)

# Grammar definition for text filter syntax with nested grouping support
TEXT_FILTER_GRAMMAR = r"""
?start: expr

// Expression with OR as lowest precedence
?expr: or_expr

// OR has lower precedence than AND
or_expr: and_expr (_OR and_expr)*

// AND (explicit or implicit via space) has higher precedence than OR
and_expr: unary_expr (_AND? unary_expr)*

// Unary expressions: negation or atom
?unary_expr: negated_expr
            | atom

negated_expr: "-" atom

// Atom: comparison, in_comparison, or parenthesized expression
?atom: in_comparison
      | comparison
      | "(" expr ")"

// IN/NOT IN comparison: field IN [...] or field NOT IN [...]
in_comparison: field _IN list_value          -> in_expr
             | field _NOT _IN list_value     -> not_in_expr

// Standard comparison: field operator value
comparison: field operator value

// Field can be simple, dotted, or metadata access
?field: metadata_field
      | dotted_field
      | simple_field

metadata_field: IDENTIFIER "[" QUOTED_STRING "]"
dotted_field: IDENTIFIER ("." IDENTIFIER)+
simple_field: IDENTIFIER

// Operators
?operator: EXACT       -> op_exact
         | LIKE        -> op_like
         | GTE         -> op_gte
         | GT          -> op_gt
         | LTE         -> op_lte
         | LT          -> op_lt

// Values (strings must be quoted)
?value: NULL           -> null_value
      | QUOTED_STRING  -> string_value
      | NUMBER         -> number_value

// List value: ["item1", "item2", ...] or []
list_value: "[" "]"                            -> empty_list
          | "[" list_item ("," list_item)* "]" -> list_items

?list_item: NULL           -> null_value
          | QUOTED_STRING  -> string_value
          | NUMBER         -> number_value

// Terminals - comparison operators
EXACT: ":"
LIKE: "~"
GT: ">"
GTE: ">="
LT: "<"
LTE: "<="

// Keywords with underscore prefix (filtered from parse tree) and higher priority
_AND.2: /AND/i
_OR.2: /OR/i
_IN.2: /IN/i
_NOT.2: /NOT/i
NULL.2: /null/i

// General terminals (lower priority)
IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
NUMBER: /-?[0-9]+(\.[0-9]+)?/
QUOTED_STRING: /"([^"\\]|\\.)*"/

%import common.WS
%ignore WS
"""


@v_args(inline=True)
class TextFilterTransformer(Transformer):
    """Transform Lark parse tree into FilterOperation objects."""

    def or_expr(self, *exprs: FilterOperation) -> FilterOperation:
        """Handle OR expression."""
        operations = list(exprs)
        if len(operations) == 1:
            return operations[0]
        return LogicalOperation(operator=FilterOperator.OR, operations=operations)

    def and_expr(self, *exprs: FilterOperation) -> FilterOperation:
        """Handle AND expression (explicit or implicit)."""
        operations = list(exprs)
        if len(operations) == 1:
            return operations[0]
        return LogicalOperation(operator=FilterOperator.AND, operations=operations)

    def negated_expr(self, inner: FilterOperation) -> LogicalOperation:
        """Handle negated expression (- prefix on comparison or group)."""
        return LogicalOperation(operator=FilterOperator.NOT, operations=[inner])

    def comparison(self, field: str, operator: FilterOperator, value: Any) -> ComparisonOperation:
        """Handle field comparison."""
        return ComparisonOperation(operator=operator, field=field, value=value)

    def in_expr(self, field: str, values: List[Any]) -> ComparisonOperation:
        """Handle IN expression: field IN [values]."""
        return ComparisonOperation(operator=FilterOperator.IN, field=field, value=values)

    def not_in_expr(self, field: str, values: List[Any]) -> ComparisonOperation:
        """Handle NOT IN expression: field NOT IN [values]."""
        return ComparisonOperation(operator=FilterOperator.NIN, field=field, value=values)

    def empty_list(self) -> List[Any]:
        """Handle empty list: []."""
        return []

    def list_items(self, *items: Any) -> List[Any]:
        """Handle non-empty list: [item1, item2, ...]."""
        return list(items)

    def simple_field(self, name: str) -> str:
        """Handle simple field name."""
        return str(name)

    def dotted_field(self, *parts: str) -> str:
        """Handle dotted field path (e.g., data.nested.key)."""
        return ".".join(str(p) for p in parts)

    def metadata_field(self, base: str, key: str) -> str:
        """Handle metadata field access (e.g., metadata["key"])."""
        unquoted_key = self._unquote_string(str(key))
        return f"{base}.{unquoted_key}"

    def op_exact(self, _token) -> FilterOperator:
        return FilterOperator.EQ

    def op_like(self, _token) -> FilterOperator:
        return FilterOperator.LIKE

    def op_gt(self, _token) -> FilterOperator:
        return FilterOperator.GT

    def op_gte(self, _token) -> FilterOperator:
        return FilterOperator.GTE

    def op_lt(self, _token) -> FilterOperator:
        return FilterOperator.LT

    def op_lte(self, _token) -> FilterOperator:
        return FilterOperator.LTE

    def null_value(self, _token) -> None:
        """Handle NULL value."""
        return None

    def string_value(self, token) -> str:
        """Handle quoted string value."""
        return self._unquote_string(str(token))

    def number_value(self, token) -> Union[int, float]:
        """Handle numeric value."""
        s = str(token)
        if "." in s:
            return float(s)
        return int(s)

    @staticmethod
    def _unquote_string(s: str) -> str:
        """Strip surrounding quotes and decode escape sequences.

        Uses ``ast.literal_eval`` to handle the full set of Python string
        escapes (``\\\\``, ``\\n``, ``\\t``, ``\\uXXXX``, etc.) consistently,
        not just ``\\"``.  Falls back to a naive strip-and-replace if the
        input is not a valid Python string literal — keeps behavior
        well-defined for edge inputs the grammar still permits.
        """
        try:
            decoded = ast.literal_eval(s)
            if isinstance(decoded, str):
                return decoded
        except (ValueError, SyntaxError):
            pass
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        return s.replace('\\"', '"')


class TextFilterParser:
    """Parser for text-based filter queries."""

    def __init__(self):
        self._parser = Lark(
            TEXT_FILTER_GRAMMAR,
            parser="lalr",
            transformer=TextFilterTransformer(),
        )

    def parse(self, query: str) -> FilterOperation:
        """Parse a filter query string into a FilterOperation."""
        if not query or not query.strip():
            raise ValueError("Filter query cannot be empty")
        return cast(FilterOperation, self._parser.parse(query.strip()))


# Module-level parser instance for convenience
_parser = TextFilterParser()


def parse_text_filter(query: str) -> FilterOperation:
    """Parse a text-based filter query string.

    Args:
        query: The filter query string

    Returns:
        A FilterOperation representing the parsed query

    Examples:
        >>> op = parse_text_filter('name:"llama"')
        >>> op = parse_text_filter('status:"active" AND amount>500')
        >>> op = parse_text_filter('currency:"usd" OR currency:"eur"')
        >>> op = parse_text_filter('-currency:"jpy"')
        >>> op = parse_text_filter('status IN ["active", "pending"]')
        >>> op = parse_text_filter('status NOT IN ["deleted"]')
    """
    return _parser.parse(query)
