# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filter paths into JSON columns other than ``data`` (e.g. ``public_metadata.key``)."""

import pytest
from nhx.core.entities.app.repository.sqlalchemy.filter import SQLAlchemyFilterRepository
from sqlalchemy import JSON, Column, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Session


class Base(DeclarativeBase):
    pass


class FakeCredential(Base):
    __tablename__ = "fake_credential"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    public_metadata = Column(JSON)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            [
                FakeCredential(id=1, name="a", public_metadata={"ids": ["ws-a"], "owner": {"team": "red"}}),
                FakeCredential(id=2, name="b", public_metadata={"ids": ["ws-b", "ws-a"], "owner": {"team": "blue"}}),
                FakeCredential(id=3, name="c", public_metadata={"ids": ["ws-c"], "owner": {"team": "red"}}),
                FakeCredential(id=4, name="d", public_metadata={}),
            ]
        )
        session.commit()
        yield session


def _ids(db: Session, condition) -> list[int]:
    return sorted(db.execute(select(FakeCredential.id).where(condition)).scalars().all())


def test_contains_on_a_path_into_a_non_data_json_column(db: Session) -> None:
    repo = SQLAlchemyFilterRepository(FakeCredential, dialect_name="sqlite")

    assert _ids(db, repo.contains("public_metadata.ids", "ws-a")) == [1, 2]
    assert _ids(db, repo.contains("public_metadata.ids", "ws-x")) == []


def test_eq_on_a_nested_path_into_a_non_data_json_column(db: Session) -> None:
    repo = SQLAlchemyFilterRepository(FakeCredential)

    assert _ids(db, repo.eq("public_metadata.owner.team", "red")) == [1, 3]


def test_a_path_into_a_non_json_column_is_rejected() -> None:
    repo = SQLAlchemyFilterRepository(FakeCredential)

    with pytest.raises(ValueError, match="does not exist"):
        repo.eq("name.first", "a")
    with pytest.raises(ValueError, match="does not exist"):
        repo.eq("missing.key", "a")
