from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="r2d1",
    version="0.1.6",
    description="Lightweight ML checkpoint courier — Cloudflare R2 storage + optional D1 metrics",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "boto3>=1.28",
        "requests>=2.28",
        "numpy>=1.23",
        "python-dotenv>=1.0",
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
