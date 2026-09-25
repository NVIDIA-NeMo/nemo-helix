# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLAlchemy implementation of FilterRepository."""

from datetime import datetime
from typing import Any, List, Optional, Set

from nhx.common.api.filter import FilterOperation, FilterRepository
from sqlalchemy import JSON, ColumnElement, DateTime, String, and_, cast, false, func, not_, or_, select
from sqlalchemy.orm import aliased


def _escape_like(value: str) -> str:
    """Escape SQL LIKE metacharacters so ``%`` and ``_`` match literally.

    ``$like`` is a case-insensitive substring (contains) test in which ``%`` and
    ``_`` are ordinary characters — the canonical contract documented and pinned
    by ``InMemoryFilterRepository.like``. The backslash escape character is
    escaped first so the escapes we add are not themselves re-escaped.
    """
    return str(value).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class SQLAlchemyFilterRepository(FilterRepository):
    """SQLAlchemy implementation of FilterRepository.

    Provides filter expression building for SQLAlchemy queries.
    Supports both PostgreSQL (JSONB) and SQLite (JSON) backends.
    """

    def __init__(self, model: Any, relationship_child_workspaces: Optional[Set[str]] = None):
        """Initialize repository with SQLAlchemy model class or alias.

        Args:
            model: SQLAlchemy model class (or aliased class) to build filters against
            relationship_child_workspaces: If set, EXISTS subqueries for parent-child relations
                only count children whose `workspace` is in this set. If None, child workspace
                is unconstrained. An empty set makes relationship EXISTS match nothing.
        """
        self.model = model
        self._relationship_child_workspaces = relationship_child_workspaces

    def _get_json_element(self, column: ColumnElement, path: List[str]) -> ColumnElement:
        """Navigate to a nested JSON element using subscript operators.

        Args:
            column: The JSON column
            path: List of keys to navigate (e.g., ['nested', 'key'])

        Returns:
            SQLAlchemy JSON element accessor
        """
        result = column
        for key in path:
            result = result[key]
        return result

    def _get_column(self, field: str) -> tuple[ColumnElement, bool]:
        """Get a column from the model by field name.

        Args:
            field: Field name to look up

        Returns:
            Tuple of (SQLAlchemy column/element, is_json) where is_json indicates
            whether the column is a JSON element accessor

        Raises:
            ValueError: If field doesn't exist on the model
        """
        # explicit field check
        if hasattr(self.model, field):
            return getattr(self.model, field), False

        # check for data access
        if field.startswith("data."):
            column = getattr(self.model, "data")
            path = field.split(".")[1:]
            return self._get_json_element(column, path), isinstance(column.type, JSON)

        raise ValueError(f"Field '{field}' does not exist on model {self.model.__name__}")

    def _coerce_value_for_column(self, column: ColumnElement, value: Any) -> Any:
        """Coerce Python types from filter inputs based on column type.

        Returns a datetime object (not a string) so that each dialect's type system
        handles formatting: PostgreSQL's psycopg2 sends it as a native TIMESTAMP,
        while SQLite's bind_processor formats it to a string with microseconds.

        Timezone info is stripped because our columns are TIMESTAMP WITHOUT TIME ZONE;
        a tz-aware datetime would cause type mismatches on PostgreSQL.
        """
        if isinstance(column.type, DateTime) and isinstance(value, str):
            return datetime.fromisoformat(value).replace(tzinfo=None)

        return value

    def _cast_json_to_raw_text(self, column: Any) -> Any:
        """Cast a JSON column element to its raw serialized text, quotes and all.

        Unlike ``_cast_json_to_text``, this keeps JSON's surrounding double quotes. Use it when the
        quotes carry meaning — e.g. matching a quote-delimited array element (``$contains``) or comparing
        against the literal ``"null"``/``"true"``/``"false"`` tokens both backends render.
        """
        return cast(column, String)

    def _cast_json_to_text(self, column: Any) -> Any:
        """Cast a JSON column element to text, trimming JSON's surrounding double quotes.

        SQLite's json_extract returns string values with quotes (e.g., '"value"'), and PostgreSQL's
        JSONB subscript also returns JSON-formatted strings. Trimming yields a consistent bare value for
        equality/substring comparison. Use ``_cast_json_to_raw_text`` when the quotes must be preserved.
        """
        return func.trim(self._cast_json_to_raw_text(column), '"')

    def _cast_json_to_numeric(self, column: Any) -> Any:
        """Cast a JSON column element to a float for numeric comparisons.

        Uses CAST(... AS FLOAT) which works on both SQLite (REAL) and PostgreSQL.
        """
        from sqlalchemy import Float

        return cast(self._cast_json_to_text(column), Float)

    def _json_comparison(self, field: str, value: Any, op: str) -> Any:
        """Build a comparison for JSON fields, using numeric cast when value is numeric."""
        column, is_json = self._get_column(field)
        if is_json:
            if isinstance(value, (int, float)):
                casted = self._cast_json_to_numeric(column)
            else:
                casted = self._cast_json_to_text(column)
                value = str(value)
            return getattr(casted, op)(value)
        return getattr(column, op)(self._coerce_value_for_column(column, value))

    def eq(self, field: str, value: Any) -> Any:
        """Equal comparison."""
        column, is_json = self._get_column(field)
        if is_json:
            # Handle None/null: match both an explicit JSON null and an absent key. A present-but-null
            # value extracts to the JSON text token "null"; a missing key extracts to SQL NULL (notably
            # on PostgreSQL, where `data->'key'` on an absent key is SQL NULL, so casting it would never
            # equal "null"). Test both so `field == None` catches missing and explicitly-null values.
            if value is None:
                return or_(self._cast_json_to_raw_text(column) == "null", column.is_(None))
            # Handle boolean values specially:
            # - SQLite stores JSON booleans as integers (0/1), json_extract returns "0" or "1"
            # - PostgreSQL stores them as "false"/"true"
            # We check both formats for cross-database compatibility
            if isinstance(value, bool):
                sqlite_value = "1" if value else "0"
                pg_value = "true" if value else "false"
                return or_(
                    self._cast_json_to_raw_text(column) == sqlite_value,
                    self._cast_json_to_raw_text(column) == pg_value,
                )
            # For string values, use _cast_json_to_text to handle quoted JSON output
            return self._cast_json_to_text(column) == str(value)
        return column == value

    def like(self, field: str, value: str) -> Any:
        """Case-insensitive substring (contains) comparison.

        ``%`` and ``_`` in ``value`` are matched literally, not as SQL wildcards,
        to agree with ``InMemoryFilterRepository.like``. Metacharacters are escaped
        and an explicit ``ESCAPE`` clause is used, which behaves the same on SQLite
        and PostgreSQL.
        """
        column, is_json = self._get_column(field)
        pattern = f"%{_escape_like(value)}%"
        if is_json:
            return self._cast_json_to_text(column).ilike(pattern, escape="\\")
        return column.ilike(pattern, escape="\\")

    def lt(self, field: str, value: Any) -> Any:
        """Less than comparison."""
        return self._json_comparison(field, value, "__lt__")

    def lte(self, field: str, value: Any) -> Any:
        """Less than or equal comparison."""
        return self._json_comparison(field, value, "__le__")

    def gt(self, field: str, value: Any) -> Any:
        """Greater than comparison."""
        return self._json_comparison(field, value, "__gt__")

    def gte(self, field: str, value: Any) -> Any:
        """Greater than or equal comparison."""
        return self._json_comparison(field, value, "__ge__")

    def in_op(self, field: str, values: List[Any]) -> Any:
        """In comparison."""
        column, is_json = self._get_column(field)
        if is_json:
            return self._cast_json_to_text(column).in_([str(v) for v in values])
        return column.in_(values)

    def nin(self, field: str, values: List[Any]) -> Any:
        """Not in comparison."""
        column, is_json = self._get_column(field)
        if is_json:
            return self._cast_json_to_text(column).not_in([str(v) for v in values])
        return column.not_in(values)

    def contains(self, field: str, value: Any) -> Any:
        """Array membership: true when the JSON array at ``field`` contains scalar ``value``.

        Portable across SQLite (JSON) and PostgreSQL (JSONB) without a dialect branch: the
        array element serializes as a quote-delimited token (e.g. ``"g1"``) in both backends'
        text rendering, so we match that token in the serialized array text. Quoting makes it
        collision-safe against prefixes (``"g1"`` does not match ``["g10"]``). ``value`` is
        coerced to text and LIKE wildcards are escaped, so only exact elements match.

        Intended for array-valued JSON fields (e.g. ``data.experiment_ids``); values are
        assumed to be JSON scalars without embedded double quotes (entity ids qualify).
        """
        column, is_json = self._get_column(field)
        if not is_json:
            raise ValueError(f"$contains requires a JSON array field, got non-JSON field '{field}'")
        needle = str(value)
        for ch in ("\\", "%", "_"):
            needle = needle.replace(ch, f"\\{ch}")
        return self._cast_json_to_raw_text(column).like(f'%"{needle}"%', escape="\\")

    def and_op(self, operations: List[Any]) -> Any:
        """Logical AND."""
        return and_(*operations)

    def or_op(self, operations: List[Any]) -> Any:
        """Logical OR."""
        return or_(*operations)

    def not_op(self, operation: Any) -> Any:
        """Logical NOT."""
        return not_(operation)

    def relationship_exists(
        self,
        target_entity_type: str,
        join_field: str,
        child_condition: Optional[FilterOperation],
        negate: bool,
    ) -> Any:
        """Build an EXISTS/NOT EXISTS subquery for a parent-child relationship."""
        child_alias = aliased(self.model)
        child_repo = SQLAlchemyFilterRepository(
            child_alias,
            relationship_child_workspaces=self._relationship_child_workspaces,
        )

        conditions = [child_alias.entity_type == target_entity_type]
        if join_field == "parent":
            conditions.append(child_alias.parent == self.model.id)
        else:
            raise NotImplementedError(f"Unsupported join_field: {join_field!r}")

        if self._relationship_child_workspaces is not None:
            if not self._relationship_child_workspaces:
                conditions.append(false())
            else:
                conditions.append(child_alias.workspace.in_(list(self._relationship_child_workspaces)))

        if child_condition is not None:
            conditions.append(child_condition.apply(child_repo))

        subq = select(child_alias.id).where(and_(*conditions)).correlate(self.model).exists()
        return ~subq if negate else subq
