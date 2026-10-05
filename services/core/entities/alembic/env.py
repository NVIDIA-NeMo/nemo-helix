# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from nhx.common.config import DatabaseConfig
from sqlalchemy import engine_from_config, pool

# Load .env if present (for local dev).
load_dotenv()

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config
# Use DatabaseConfig only when URL wasn't already set (e.g. by startup or tests).
if not config.get_main_option("sqlalchemy.url"):
    config.set_main_option("sqlalchemy.url", DatabaseConfig().sqlalchemy_database_url())

# Interpret the config file for Python logging.
# When running inside the NeMo Helix, the root logger already has a structlog
# handler (ProcessorFormatter). Skip fileConfig so we don't add alembic.ini's
# generic handler; then Alembic logs propagate to root and use the platform format.
# When running Alembic from the CLI, root has no such handler, so we apply fileConfig.
# When running under pytest, skip fileConfig so we don't replace root's handlers
# (fileConfig removes existing root handlers), which would break caplog and other
# log-capture tests.
_root = logging.getLogger()
_in_platform = any(
    h.formatter is not None and type(h.formatter).__name__ == "ProcessorFormatter" for h in _root.handlers
)
_in_pytest = "pytest" in sys.modules
if config.config_file_name is not None and not _in_platform and not _in_pytest:
    fileConfig(config.config_file_name, disable_existing_loggers=False)
elif _in_platform or _in_pytest:
    logging.getLogger("alembic").setLevel(logging.INFO)

# For autogenerate: use Entities Base.metadata; import models so all tables are registered.
import nhx.core.entities.app.repository.sqlalchemy.models  # noqa: E402, F401
from nhx.core.entities.app.repository.sqlalchemy.base import Base  # noqa: E402

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
