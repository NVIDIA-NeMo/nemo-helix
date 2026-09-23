# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-generation transform: inject exist_ok into generated create() methods.

Walks all resource files under the SDK's resources/ directory and finds classes
that have both a `create()` and `retrieve()` method with a `name` parameter.
For each eligible class, the transformer:

1. Adds an `exist_ok: bool = False` parameter to create().
2. Wraps the return statement in a try/except ConflictError block that falls
   back to self.retrieve() when exist_ok is True.
3. Adds the ConflictError import to modified files.

This runs as part of the post-generation pipeline so the modification survives
SDK regeneration.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import libcst as cst
import typer
from nemo_helix_sdk_tools.sdk.core.common import get_sdk_info

logger = logging.getLogger(__name__)

SKIP_RESOURCES = {
    "jobs",
    "completions",
    "otlp",
    "members",
    "adapters",
    "steps",
    "tasks",
    "results",
    "gateway",
    "benchmark_jobs",
    "metric_jobs",
    "secrets",
}


def _has_param(func: cst.FunctionDef, param_name: str) -> bool:
    """Check whether a FunctionDef has a keyword-only parameter with the given name."""
    for param in func.params.params + func.params.kwonly_params:
        if isinstance(param.name, cst.Name) and param.name.value == param_name:
            return True
    return False


def _is_async(func: cst.FunctionDef) -> bool:
    return func.asynchronous is not None


def _is_overload(func: cst.FunctionDef) -> bool:
    """Check if a function is decorated with @overload."""
    for dec in func.decorators:
        node = dec.decorator
        if isinstance(node, cst.Name) and node.value == "overload":
            return True
        if isinstance(node, cst.Attribute) and node.attr.value == "overload":
            return True
    return False


def _get_method_names(class_node: cst.ClassDef) -> dict[str, cst.FunctionDef]:
    """Return a mapping of method name -> FunctionDef for a class body."""
    methods: dict[str, cst.FunctionDef] = {}
    for stmt in class_node.body.body:
        if isinstance(stmt, cst.FunctionDef):
            methods[stmt.name.value] = stmt
    return methods


def _build_exist_ok_param() -> cst.Param:
    """Build the ``exist_ok: bool = False`` parameter node."""
    return cst.Param(
        name=cst.Name("exist_ok"),
        annotation=cst.Annotation(annotation=cst.Name("bool")),
        default=cst.Name("False"),
    )


def _build_retrieve_call(*, has_workspace: bool, is_async: bool) -> cst.BaseExpression:
    """Build ``self.retrieve(name=name)`` or ``self.retrieve(name=name, workspace=workspace)``."""
    args = [cst.Arg(keyword=cst.Name("name"), value=cst.Name("name"))]
    if has_workspace:
        args.append(cst.Arg(keyword=cst.Name("workspace"), value=cst.Name("workspace")))

    call = cst.Call(
        func=cst.Attribute(value=cst.Name("self"), attr=cst.Name("retrieve")),
        args=args,
    )
    if is_async:
        return cst.Await(expression=call)
    return call


EXIST_OK_DOC = "exist_ok: Do not raise an error if the resource already exists. Returns the existing resource."


def _is_docstring(stmt: cst.BaseStatement) -> bool:
    """Check if a statement is a docstring (a simple statement containing only an Expr with a string)."""
    if not isinstance(stmt, cst.SimpleStatementLine):
        return False
    if len(stmt.body) != 1:
        return False
    expr = stmt.body[0]
    if not isinstance(expr, cst.Expr):
        return False
    return isinstance(expr.value, (cst.SimpleString, cst.FormattedString, cst.ConcatenatedString))


def _inject_exist_ok_into_docstring(stmt: cst.SimpleStatementLine) -> cst.SimpleStatementLine:
    """Add the exist_ok arg description into a docstring's Args section."""
    expr = stmt.body[0]
    assert isinstance(expr, cst.Expr)
    string_node = expr.value

    if not isinstance(string_node, cst.SimpleString):
        logger.warning("Docstring is not a SimpleString, skipping exist_ok doc injection.")
        return stmt

    raw = string_node.value

    if "exist_ok" in raw:
        return stmt

    extra_headers_match = re.search(r"(\n(\s+)extra_headers:)", raw)
    if extra_headers_match:
        indent = extra_headers_match.group(2)
        insertion = f"\n\n{indent}{EXIST_OK_DOC}\n"
        new_raw = raw[: extra_headers_match.start()] + insertion + raw[extra_headers_match.start() :]
    else:
        closing_quote = '"""' if '"""' in raw else "'''" if "'''" in raw else None
        if closing_quote is None:
            return stmt
        last_quote_idx = raw.rfind(closing_quote)
        if last_quote_idx <= 0:
            return stmt
        new_raw = raw[:last_quote_idx] + f"\n      {EXIST_OK_DOC}\n    " + raw[last_quote_idx:]

    new_string = string_node.with_changes(value=new_raw)
    new_expr = expr.with_changes(value=new_string)
    return stmt.with_changes(body=[new_expr])


def _wrap_return_in_try_except(
    body: cst.BaseSuite,
    *,
    has_workspace: bool,
    is_async: bool,
) -> cst.BaseSuite:
    """Wrap the function body so the original return is inside a try block.

    The docstring is preserved before the try block. Produces::

        \"\"\"Original docstring.\"\"\"
        try:
            <rest of original body>
        except ConflictError:
            if not exist_ok:
                raise
            return self.retrieve(name=name, ...)
    """
    retrieve_call = _build_retrieve_call(has_workspace=has_workspace, is_async=is_async)

    except_body = cst.IndentedBlock(
        body=[
            cst.If(
                test=cst.UnaryOperation(operator=cst.Not(), expression=cst.Name("exist_ok")),
                body=cst.IndentedBlock(
                    body=[
                        cst.SimpleStatementLine(body=[cst.Raise()]),
                    ]
                ),
                leading_lines=[],
            ),
            cst.SimpleStatementLine(body=[cst.Return(value=retrieve_call)]),
        ],
    )

    handler = cst.ExceptHandler(
        type=cst.Name("ConflictError"),
        body=except_body,
    )

    if not isinstance(body, cst.IndentedBlock):
        try_body = body
        prefix: list[cst.BaseStatement] = []
    else:
        stmts = list(body.body)
        prefix = []
        while stmts and _is_docstring(stmts[0]):
            prefix.append(_inject_exist_ok_into_docstring(stmts.pop(0)))
        try_body = body.with_changes(body=stmts)

    try_stmt = cst.Try(
        body=try_body,
        handlers=[handler],
    )

    return cst.IndentedBlock(body=[*prefix, try_stmt])


def _insert_param_before_extras(func: cst.FunctionDef, new_param: cst.Param) -> cst.FunctionDef:
    """Insert a parameter right before the extra_headers / extra_query group.

    The generated code always has these four trailing params:
        extra_headers, extra_query, extra_body, timeout
    We insert exist_ok just before that block, with a trailing comma so
    it renders on its own line.
    """
    extra_names = {"extra_headers", "extra_query", "extra_body", "timeout"}
    kwonly = list(func.params.kwonly_params)

    insert_idx = len(kwonly)
    for i, param in enumerate(kwonly):
        if isinstance(param.name, cst.Name) and param.name.value in extra_names:
            insert_idx = i
            break

    new_param = new_param.with_changes(
        comma=cst.Comma(
            whitespace_after=cst.ParenthesizedWhitespace(
                first_line=cst.TrailingWhitespace(newline=cst.Newline()),
                indent=True,
                last_line=cst.SimpleWhitespace("    "),
            ),
        ),
    )
    kwonly.insert(insert_idx, new_param)
    return func.with_changes(params=func.params.with_changes(kwonly_params=kwonly))


class _ExistOkInjector(cst.CSTTransformer):
    """Visit each class; if it has both create() and retrieve(name=...), transform create()."""

    def __init__(self) -> None:
        self.modified = False
        self._eligible_classes: set[str] = set()
        self._retrieve_has_workspace: dict[str, bool] = {}
        self._current_class: str | None = None
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        methods = _get_method_names(node)
        class_name = node.name.value

        create_fn = methods.get("create")
        retrieve_fn = methods.get("retrieve")

        if create_fn is None or retrieve_fn is None:
            return True

        if not _has_param(create_fn, "name"):
            return True

        self._eligible_classes.add(class_name)
        self._retrieve_has_workspace[class_name] = _has_param(retrieve_fn, "workspace")
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if updated_node.name.value != "create":
            return updated_node

        # Find enclosing class — we track eligibility by class name during visit
        # libcst doesn't give us the parent directly, but we recorded eligible classes
        # during visit_ClassDef. We rely on the fact that we only transform create()
        # methods inside eligible classes.
        # The transformer visits in order, so we use a stack approach.
        if not self._current_class or self._current_class not in self._eligible_classes:
            return updated_node

        if _has_param(updated_node, "exist_ok"):
            return updated_node

        has_workspace = self._retrieve_has_workspace.get(self._current_class, True)
        async_fn = _is_async(updated_node)

        updated_node = _insert_param_before_extras(updated_node, _build_exist_ok_param())

        if not _is_overload(updated_node):
            updated_node = updated_node.with_changes(
                body=_wrap_return_in_try_except(
                    updated_node.body,
                    has_workspace=has_workspace,
                    is_async=async_fn,
                ),
            )

        self.modified = True
        return updated_node

    def visit_ClassDef_body(self, node: cst.ClassDef) -> None:
        self._class_stack.append(node.name.value)
        self._current_class = node.name.value

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        self._class_stack.pop()
        self._current_class = self._class_stack[-1] if self._class_stack else None
        return updated_node


class _ConflictErrorImportAdder(cst.CSTTransformer):
    """Add ``from ..._exceptions import ConflictError`` if not already present.

    Handles varying relative import depths by inspecting existing imports from
    the ``_exceptions`` module.
    """

    def __init__(self, file_path: str = "<unknown>") -> None:
        self._has_conflict_import = False
        self._relative_dots: tuple[cst.Dot, ...] | None = None
        self._file_path = file_path

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if isinstance(node.names, (list, tuple)):
            for alias in node.names:
                name = alias.name.value if isinstance(alias.name, cst.Name) else None
                if name == "ConflictError":
                    self._has_conflict_import = True

        # Infer the relative import depth from existing relative imports
        # to private modules (e.g., from ..._types import ...).
        if not node.relative:
            return
        module = node.module
        if module and isinstance(module, cst.Attribute):
            full = _module_to_str(module)
            if full and full.startswith("_"):
                self._relative_dots = node.relative
        elif module and isinstance(module, cst.Name) and module.value.startswith("_"):
            self._relative_dots = node.relative

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        if self._has_conflict_import:
            return updated_node

        if self._relative_dots is None:
            logger.warning(
                "Could not infer relative import depth for ConflictError in %s. "
                "No existing relative import from a private module (e.g., from ..._types import ...) was found. "
                "Skipping ConflictError import injection for this file.",
                self._file_path,
            )
            return updated_node

        import_stmt = cst.SimpleStatementLine(
            body=[
                cst.ImportFrom(
                    relative=self._relative_dots,
                    module=cst.Name("_exceptions"),
                    names=[cst.ImportAlias(name=cst.Name("ConflictError"))],
                ),
            ]
        )

        body = list(updated_node.body)
        insert_idx = 0
        for i, stmt in enumerate(body):
            if isinstance(stmt, cst.SimpleStatementLine) and any(
                isinstance(s, (cst.Import, cst.ImportFrom)) for s in stmt.body
            ):
                insert_idx = i + 1

        body.insert(insert_idx, import_stmt)
        return updated_node.with_changes(body=body)


def _module_to_str(node: cst.BaseExpression) -> str | None:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        parent = _module_to_str(node.value)
        if parent:
            return f"{parent}.{node.attr.value}"
    return None


def _should_process_file(file_path: Path) -> bool:
    """Decide whether a resource file should be processed."""
    parts = set(file_path.parts)
    if parts & SKIP_RESOURCES:
        return False
    if file_path.name.startswith("_"):
        return False
    return True


def inject_into_methods(sdk_dir: Path) -> int:
    """Walk resource files and inject exist_ok into eligible create() methods.

    Returns the number of files modified.
    """
    resources_dir = sdk_dir / "src" / "nemo_helix" / "resources"
    if not resources_dir.exists():
        typer.echo(f"  Resources directory not found: {resources_dir}")
        return 0

    modified_count = 0

    for py_file in sorted(resources_dir.rglob("*.py")):
        if not _should_process_file(py_file.relative_to(resources_dir)):
            continue

        source = py_file.read_text(encoding="utf-8")
        tree = cst.parse_module(source)

        injector = _ExistOkInjector()
        modified_tree = tree.visit(injector)

        needs_import = injector.modified or "ConflictError" in source
        if needs_import:
            import_adder = _ConflictErrorImportAdder(file_path=str(py_file.relative_to(sdk_dir)))
            modified_tree = modified_tree.visit(import_adder)

        if modified_tree.code == source:
            continue

        py_file.write_text(modified_tree.code, encoding="utf-8")
        rel_path = py_file.relative_to(sdk_dir)
        typer.echo(f"  - Injected exist_ok into {rel_path}")
        modified_count += 1

    return modified_count


def inject_exist_ok() -> None:
    """Inject exist_ok parameter into generated create() methods."""
    sdk_info = get_sdk_info()

    typer.echo("Injecting exist_ok into create() methods...")
    count = inject_into_methods(sdk_info.sdk_dir)
    typer.echo(f"  Modified {count} files.")
    typer.echo("exist_ok injection completed!")
