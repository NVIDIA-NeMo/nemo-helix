# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLAlchemy implementation of FilterRepository."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from nemo_helix_plugin.filter_ops import ElemMatchScalar
from nhx.common.api.filter import FilterOperation, FilterOperator, FilterRepository
from sqlalchemy import (
    JSON,
    ColumnElement,
    DateTime,
    String,
    and_,
    case,
    cast,
    false,
    func,
    literal,
    literal_column,
    not_,
    or_,
    select,
    type_coerce,
)
from sqlalchemy.orm import aliased


class SQLAlchemyFilterRepository(FilterRepository):
    """SQLAlchemy implementation of FilterRepository.

    Provides filter expression building for SQLAlchemy queries.
    Supports both PostgreSQL (JSONB) and SQLite (JSON) backends.
    """

    def __init__(
        self,
        model: Any,
        relationship_child_workspaces: Optional[Set[str]] = None,
        dialect_name: Optional[str] = None,
    ):
        """Initialize repository with SQLAlchemy model class or alias.

        Args:
            model: SQLAlchemy model class (or aliased class) to build filters against
            relationship_child_workspaces: If set, EXISTS subqueries for parent-child relations
                only count children whose `workspace` is in this set. If None, child workspace
                is unconstrained. An empty set makes relationship EXISTS match nothing.
            dialect_name: Database dialect (``"sqlite"`` or ``"postgresql"``). Required only by
                operators whose SQL differs per backend (``$elemMatch``, ``$containsPrefix``).
        """
        self.model = model
        self._relationship_child_workspaces = relationship_child_workspaces
        self._dialect_name = dialect_name

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

    @staticmethod
    def _escape_like(text: str) -> str:
        for ch in ("\\", "%", "_"):
            text = text.replace(ch, f"\\{ch}")
        return text

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
            return self._json_eq(column, value)
        return column == value

    def _json_eq(self, element: Any, value: Any) -> Any:
        # Handle None/null: match both an explicit JSON null and an absent key. A present-but-null
        # value extracts to the JSON text token "null"; a missing key extracts to SQL NULL (notably
        # on PostgreSQL, where `data->'key'` on an absent key is SQL NULL, so casting it would never
        # equal "null"). Test both so `field == None` catches missing and explicitly-null values.
        if value is None:
            return or_(self._cast_json_to_raw_text(element) == "null", element.is_(None))
        # Handle boolean values specially:
        # - SQLite stores JSON booleans as integers (0/1), json_extract returns "0" or "1"
        # - PostgreSQL stores them as "false"/"true"
        # We check both formats for cross-database compatibility
        if isinstance(value, bool):
            sqlite_value = "1" if value else "0"
            pg_value = "true" if value else "false"
            return or_(
                self._cast_json_to_raw_text(element) == sqlite_value,
                self._cast_json_to_raw_text(element) == pg_value,
            )
        # For string values, use _cast_json_to_text to handle quoted JSON output
        return self._cast_json_to_text(element) == str(value)

    def like(self, field: str, value: str) -> Any:
        """Like/contains comparison."""
        column, is_json = self._get_column(field)
        if is_json:
            return self._cast_json_to_text(column).ilike(f"%{value}%")
        return column.ilike(f"%{value}%")

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
        return self._cast_json_to_raw_text(column).like(f'%"{self._escape_like(str(value))}"%', escape="\\")

    def contains_prefix(self, field: str, prefix: str) -> Any:
        array, is_json = self._get_column(field)
        if not is_json:
            raise ValueError(f"$containsPrefix requires a JSON array field, got non-JSON field '{field}'")
        elements = self._array_elements(array, FilterOperator.CONTAINS_PREFIX)
        pattern = f"{self._escape_like(prefix)}%"
        if self._dialect_name == "sqlite":
            match = and_(elements.c.type == "text", elements.c.value.like(pattern, escape="\\"))
        else:
            text = elements.c.value.op("#>>")(literal_column("'{}'::text[]"))
            match = and_(func.json_typeof(elements.c.value) == "string", text.like(pattern, escape="\\"))
        return select(literal(1)).select_from(elements).where(match).exists()

    def has_key(self, field: str, key: str) -> Any:
        column, is_json = self._get_column(field)
        if not is_json:
            raise ValueError(f"$hasKey requires a JSON object field, got non-JSON field '{field}'")
        return not_(self._json_eq(column[key], None))

    def _array_elements(self, array: Any, operator: FilterOperator) -> Any:
        """The elements of the JSON array ``array`` as a table with a ``value`` column (and ``type`` on SQLite)."""
        # A non-array must expand to nothing: PostgreSQL raises on it, SQLite would iterate an object's members.
        if self._dialect_name == "sqlite":
            only_array = case((func.json_type(array) == "array", array))
            return func.json_each(only_array).table_valued("value", "type")
        if self._dialect_name == "postgresql":
            only_array = case((func.json_typeof(array) == "array", array))
            return func.json_array_elements(only_array).table_valued("value")
        raise ValueError(f"{operator.value} is not supported for database dialect {self._dialect_name!r}")

    def elem_match(self, field: str, criteria: Dict[str, ElemMatchScalar]) -> Any:
        array, is_json = self._get_column(field)
        if not is_json:
            raise ValueError(f"$elemMatch requires a JSON array field, got non-JSON field '{field}'")
        elements = self._array_elements(array, FilterOperator.ELEM_MATCH)
        element = type_coerce(elements.c.value, JSON)
        matches = [self._json_eq(element[key], value) for key, value in criteria.items()]
        return select(literal(1)).select_from(elements).where(*matches).exists()

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
            dialect_name=self._dialect_name,
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
