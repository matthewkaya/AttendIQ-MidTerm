"""Deployment files stay aligned with the assignment's Space requirements."""

import re
from pathlib import Path

import gradio
import sklearn

ROOT = Path(__file__).resolve().parents[1]


def _requirement_version(requirements: str, package: str) -> str:
    match = re.search(rf"^{re.escape(package)}==([^\s;]+)$", requirements, re.MULTILINE)
    assert match is not None, f"{package} must have an exact version pin"
    return match.group(1)


def test_space_readme_settings_match_runtime_and_app():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = readme.split("---", maxsplit=2)[1]
    settings = dict(line.split(": ", maxsplit=1) for line in block.strip().splitlines())

    assert settings["sdk"] == "gradio"
    assert settings["sdk_version"] == gradio.__version__
    assert settings["python_version"] == "3.12"
    assert settings["app_file"] == "app.py"
    assert (ROOT / settings["app_file"]).is_file()


def test_runtime_pins_match_installed_and_exclude_gradio():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    dev_requirements = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")

    assert _requirement_version(requirements, "scikit-learn") == sklearn.__version__
    assert _requirement_version(requirements, "pandas")
    assert _requirement_version(requirements, "joblib")
    assert not re.search(r"^gradio(?:[<=>!~]|$)", requirements, re.MULTILINE)
    assert "-r requirements.txt" in dev_requirements
    assert _requirement_version(dev_requirements, "gradio") == gradio.__version__
    assert _requirement_version(dev_requirements, "pytest")
    assert _requirement_version(dev_requirements, "ruff")


def test_python_312_is_selected_in_environment_and_package_metadata():
    environment = (ROOT / "environment.yml").read_text(encoding="utf-8")
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert re.search(r"^\s+- python=3\.12\s*$", environment, re.MULTILINE)
    assert 'requires-python = ">=3.12,<3.13"' in project


def test_space_gpu_placeholder_is_guarded_and_decorated():
    source = (ROOT / "app.py").read_text(encoding="utf-8")

    assert (
        "try:\n    import spaces\n\n    @spaces.GPU\n    def _spaces_gpu_placeholder"
        in source
    )
    assert "except ImportError:" in source
    assert "def _spaces_gpu_placeholder() -> None:" in source


def test_model_artifacts_are_not_ignored():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert not re.search(
        r"^models/(?:\*|.*\.(?:joblib|pkl|json))", gitignore, re.MULTILINE
    )
