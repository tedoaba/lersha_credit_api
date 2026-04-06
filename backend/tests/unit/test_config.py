"""Unit tests for backend/config/config.py.

Verifies startup validation behaviour without requiring real environment
variables — uses monkeypatching and temporary files.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _fresh_config(tmp_hparams: Path | None = None, monkeypatch=None):
    """Import a fresh Config() instance with controlled environment and paths.

    Removes any cached backend.config.config module to force re-import.
    If ``tmp_hparams`` is given, patches BASE_DIR so the config will look
    for hyperparams.yaml in a temporary directory.
    """
    # Ensure required env vars are present (values are fake — only structure tested)
    env_overrides = {
        "API_KEY": "test-key",
        "OLLAMA_MODEL_NAME": "test-model",
        "OLLAMA_HOST": "http://localhost:11434",
        "DB_URI": "sqlite:///:memory:",
    }
    if monkeypatch:
        for key, value in env_overrides.items():
            monkeypatch.setenv(key, value)

    # Clear cached module to force re-import
    for mod in list(sys.modules.keys()):
        if "backend.config" in mod:
            del sys.modules[mod]

    if tmp_hparams is not None:
        with patch("backend.config.config.BASE_DIR", tmp_hparams.parent.parent):
            import backend.config.config as cfg_module  # noqa: PLC0415

            return cfg_module.Config()
    else:
        import backend.config.config as cfg_module  # noqa: PLC0415

        return cfg_module.Config()


class TestConfigHyperparamsGuard:
    """Tests for the FileNotFoundError guard on hyperparams.yaml."""

    def test_config_raises_on_missing_hyperparams_yaml(self, tmp_path, monkeypatch):
        """Config() MUST raise FileNotFoundError when hyperparams.yaml is absent.

        Creates a temporary directory tree that mimics the project structure
        but does NOT include hyperparams.yaml, then asserts Config() raises.
        """
        # Build fake project tree: tmp/backend/config/ (no hyperparams.yaml inside)
        fake_backend_config = tmp_path / "backend" / "config"
        fake_backend_config.mkdir(parents=True)

        env_overrides = {
            "API_KEY": "test-key",
            "GEMINI_MODEL": "gemini-1.5-pro",
            "GEMINI_API_KEY": "test-gemini-key",
            "DB_URI": "sqlite:///:memory:",
        }
        for key, value in env_overrides.items():
            monkeypatch.setenv(key, value)

        # Clear cached modules
        for mod in list(sys.modules.keys()):
            if "backend.config" in mod:
                del sys.modules[mod]

        # Patch BASE_DIR to point to tmp_path so the absent yaml triggers the guard
        with patch("backend.config.config.BASE_DIR", tmp_path):
            import backend.config.config as cfg_module  # noqa: PLC0415

            with pytest.raises(FileNotFoundError, match="hyperparams.yaml"):
                cfg_module.Config()

    def test_config_loads_successfully_when_yaml_present(self, tmp_path, monkeypatch):
        """Config() loads hyperparams correctly when the YAML file exists."""
        # Build proper fake tree with hyperparams.yaml
        fake_config_dir = tmp_path / "backend" / "config"
        fake_config_dir.mkdir(parents=True)
        yaml_content = "inference:\n  default_batch_size: 10\nmodels:\n  active:\n    - xgboost\n"
        (fake_config_dir / "hyperparams.yaml").write_text(yaml_content, encoding="utf-8")

        env_overrides = {
            "API_KEY": "test-key",
            "GEMINI_MODEL": "gemini-1.5-pro",
            "GEMINI_API_KEY": "test-gemini-key",
            "DB_URI": "sqlite:///:memory:",
        }
        for key, value in env_overrides.items():
            monkeypatch.setenv(key, value)

        for mod in list(sys.modules.keys()):
            if "backend.config" in mod:
                del sys.modules[mod]

        with patch("backend.config.config.BASE_DIR", tmp_path):
            import backend.config.config as cfg_module  # noqa: PLC0415

            cfg = cfg_module.Config()

        assert cfg.hyperparams["inference"]["default_batch_size"] == 10
        assert "xgboost" in cfg.hyperparams["models"]["active"]
