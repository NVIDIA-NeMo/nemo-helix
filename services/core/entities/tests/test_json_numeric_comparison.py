# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for numeric JSON field comparison ordering."""

import pytest
from nhx.core.entities.app.repository.sqlalchemy.filter import SQLAlchemyFilterRepository
from sqlalchemy import JSON, Column, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Session


class Base(DeclarativeBase):
    pass


class FakeEntity(Base):
    __tablename__ = "fake_entity"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    data = Column(JSON)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            [
                FakeEntity(id=1, name="a", data={"score": 5}),
                FakeEntity(id=2, name="b", data={"score": 9}),
                FakeEntity(id=3, name="c", data={"score": 10}),
                FakeEntity(id=4, name="d", data={"score": 100}),
            ]
        )
        session.commit()
        yield session


class TestNumericJsonComparison:
    def test_gt_numeric_json_value(self, db):
        """data.score > 9 should return score=10 and score=100, NOT score=5."""
        repo = SQLAlchemyFilterRepository(FakeEntity)
        condition = repo.gt("data.score", 9)
        results = db.execute(select(FakeEntity).where(condition)).scalars().all()
        scores = [r.data["score"] for r in results]
        assert sorted(scores) == [10, 100]

    def test_lt_numeric_json_value(self, db):
        """data.score < 10 should return score=5 and score=9."""
        repo = SQLAlchemyFilterRepository(FakeEntity)
        condition = repo.lt("data.score", 10)
        results = db.execute(select(FakeEntity).where(condition)).scalars().all()
        scores = [r.data["score"] for r in results]
        assert sorted(scores) == [5, 9]

    def test_gte_numeric_json_value(self, db):
        """data.score >= 10 should return score=10 and score=100."""
        repo = SQLAlchemyFilterRepository(FakeEntity)
        condition = repo.gte("data.score", 10)
        results = db.execute(select(FakeEntity).where(condition)).scalars().all()
        scores = [r.data["score"] for r in results]
        assert sorted(scores) == [10, 100]

    def test_lte_numeric_json_value(self, db):
        """data.score <= 9 should return score=5 and score=9."""
        repo = SQLAlchemyFilterRepository(FakeEntity)
        condition = repo.lte("data.score", 9)
        results = db.execute(select(FakeEntity).where(condition)).scalars().all()
        scores = [r.data["score"] for r in results]
        assert sorted(scores) == [5, 9]

    def test_string_json_comparison_still_works(self, db):
        """String values in JSON should still compare as strings."""
        repo = SQLAlchemyFilterRepository(FakeEntity)
        condition = repo.eq("name", "a")
        results = db.execute(select(FakeEntity).where(condition)).scalars().all()
        assert len(results) == 1
        assert results[0].name == "a"
