"""Offline tests for finagent.config.

Definition of "this works":
    1. A set variable is returned, stripped of stray spaces.
    2. A missing or blank variable raises RuntimeError naming the variable.
    3. A .env file in the current folder is picked up.
    4. A real environment variable beats the .env file.
    5. The step cap and token limit are sane positive integers.
"""

import pytest

from finagent import config


def test_require_env_returns_value(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "  abc123  ")
    assert config.require_env("FAKE_KEY") == "abc123"


def test_require_env_missing_raises_and_names_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("FAKE_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FAKE_KEY"):
        config.require_env("FAKE_KEY")


def test_require_env_blank_counts_as_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FAKE_KEY", "   ")
    with pytest.raises(RuntimeError, match="FAKE_KEY"):
        config.require_env("FAKE_KEY")


def test_env_file_in_cwd_is_loaded(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("FAKE_FROM_FILE=from_file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FAKE_FROM_FILE", raising=False)
    assert config.require_env("FAKE_FROM_FILE") == "from_file"


def test_real_environment_beats_env_file(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("FAKE_BOTH=from_file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FAKE_BOTH", "from_shell")
    assert config.require_env("FAKE_BOTH") == "from_shell"


def test_limits_are_sane():
    assert isinstance(config.MAX_STEPS, int) and config.MAX_STEPS >= 6
    assert isinstance(config.MAX_TOKENS, int) and config.MAX_TOKENS > 0
    assert config.BASE_URL.startswith("https://")