"""Minimal setup so `pip install -e .` enables `import fairness_lab` from anywhere."""
from setuptools import setup, find_packages

setup(
    name="fairness_lab",
    version="4.0.0",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "lightgbm>=4.0.0",
        "sdv>=1.12.0",
    ],
)
