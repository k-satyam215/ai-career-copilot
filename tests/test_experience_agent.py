"""Tests for experience_agent."""
import pytest
from agents.experience_agent import (
    experience_agent,
    EXPERIENCE_INDICATORS,
    POINTS_PER_INDICATOR,
    MAX_EXPERIENCE_SCORE,
)


def make_state(text):
    return {"resume_chunks": [text], "full_resume_chunks": [text]}


def test_project_indicator():
    state = make_state("worked on a project for client")
    result = experience_agent(state)
    assert result["experience_score"] >= POINTS_PER_INDICATOR


def test_built_indicator():
    state = make_state("built an api for internal use")
    result = experience_agent(state)
    assert result["experience_score"] >= POINTS_PER_INDICATOR


def test_implemented_indicator():
    state = make_state("implemented a caching system")
    result = experience_agent(state)
    assert result["experience_score"] >= POINTS_PER_INDICATOR


def test_developed_indicator():
    state = make_state("developed a mobile application")
    result = experience_agent(state)
    assert result["experience_score"] >= POINTS_PER_INDICATOR


def test_designed_indicator():
    state = make_state("designed the database schema")
    result = experience_agent(state)
    assert result["experience_score"] >= POINTS_PER_INDICATOR


def test_system_indicator():
    state = make_state("created a distributed system")
    result = experience_agent(state)
    assert result["experience_score"] >= POINTS_PER_INDICATOR


def test_pipeline_indicator():
    state = make_state("built a data pipeline for etl")
    result = experience_agent(state)
    assert result["experience_score"] >= POINTS_PER_INDICATOR


def test_application_indicator():
    state = make_state("built an application for users")
    result = experience_agent(state)
    assert result["experience_score"] >= POINTS_PER_INDICATOR


def test_empty_text_zero():
    state = make_state("")
    result = experience_agent(state)
    assert result["experience_score"] == 0


def test_score_capped():
    text = " ".join(EXPERIENCE_INDICATORS * 5)
    state = make_state(text)
    result = experience_agent(state)
    assert result["experience_score"] <= MAX_EXPERIENCE_SCORE


def test_all_indicators_accumulate():
    text = " ".join(EXPERIENCE_INDICATORS)
    state = make_state(text)
    result = experience_agent(state)
    expected = min(len(EXPERIENCE_INDICATORS) * POINTS_PER_INDICATOR, MAX_EXPERIENCE_SCORE)
    assert result["experience_score"] == expected


def test_case_insensitive():
    state = make_state("BUILT a PROJECT using a PIPELINE")
    result = experience_agent(state)
    assert result["experience_score"] >= POINTS_PER_INDICATOR * 3


def test_state_returned():
    state = make_state("built a system")
    result = experience_agent(state)
    assert result is state
