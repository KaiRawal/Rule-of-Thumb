"""Packaging regression tests for the extras split (ToDo 26)."""

import re
import sys
from pathlib import Path

import pytest


def _pyproject_text():
    return (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()


def _section(text, name):
    pattern = re.compile(r"\[project\.optional-dependencies\](.*?)(?=\n\[|\Z)", re.DOTALL)
    block = pattern.search(text).group(1)
    item = re.search(rf"^{name} = \[(.*?)\]", block, re.MULTILINE | re.DOTALL).group(1).lower()
    return item


def test_base_dependencies_minimal():
    text = _pyproject_text()
    base = re.search(r"^dependencies = \[(.*?)\]", text, re.MULTILINE | re.DOTALL).group(1).lower()
    assert "numpy" in base and "torch" in base
    for heavy in ("torchvision", "transformers", "matplotlib", "seaborn", "wordcloud", "pillow", "shap", "captum"):
        assert heavy not in base


def test_extras_cover_modalities():
    text = _pyproject_text().lower()
    assert "seaborn" not in text
    assert "transformers" in _section(text, "text")
    image = _section(text, "image")
    assert "torchvision" in image and "pillow" in image
    plot = _section(text, "plot")
    for dep in ("matplotlib", "wordcloud", "shap"):
        assert dep in plot


def test_plot_missing_matplotlib_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    monkeypatch.setitem(sys.modules, "matplotlib.colors", None)
    for mod in [m for m in sys.modules if m == "ruleofthumb.plot" or m.startswith("ruleofthumb.plot.")]:
        del sys.modules[mod]
    with pytest.raises(ImportError, match=r"ruleofthumb-rot\[plot\]"):
        import ruleofthumb.plot  # noqa: F401
