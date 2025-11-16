#!/usr/bin/env python3
"""
Packaging metadata for the LLM training project.

This file defines the setuptools configuration so the project can be installed
in editable or regular mode (e.g., `pip install -e .`).
"""

from pathlib import Path
import platform
from setuptools import find_packages, setup

REPO_ROOT = Path(__file__).parent


def _read_requirements() -> list[str]:
    """
    Read the OS-specific pip-compile lockfile (falls back to requirements.in).
    This keeps `pip install -e .` consistent with the curated dependency sets.
    """
    system = platform.system().lower()
    if system == "darwin":
        req_file = REPO_ROOT / "requirements-mac.txt"
    else:
        req_file = REPO_ROOT / "requirements-linux-cuda.txt"

    if not req_file.exists():
        # Fallback for environments (e.g., Windows) where no lockfile exists yet
        req_file = REPO_ROOT / "requirements.in"
        if not req_file.exists():
            return []

    requirements: list[str] = []
    for line in req_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)
    return requirements


setup(
    name="llm-trainer",
    version="0.1.0",
    description="Modular framework for LLM fine-tuning with QLoRA (SFT, instruction tuning, reward modeling, DPO).",
    author="Weiliang Jiao",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=_read_requirements(),
    python_requires=">=3.10",
)
