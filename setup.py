from setuptools import setup, find_packages
from pathlib import Path

ROOT = Path(__file__).parent
README = ROOT / "README.md"

setup(
    name="r2d1",
    version="0.1.4",
    author="SparseTrace",
    description="Lightweight ML experiment tracker — Cloudflare R2 checkpoints + D1 metrics",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "boto3>=1.28",
        "requests>=2.28",
        "numpy>=1.23",
    ],
    extras_require={
        "torch": ["torch"],
        "jax": ["jax", "jaxlib"],
        "dev": ["build", "twine"],
    },
    python_requires=">=3.9",
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
