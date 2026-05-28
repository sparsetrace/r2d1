from setuptools import setup, find_packages

setup(
    name="r2d1",
    version="0.1.0",
    description="Lightweight ML experiment tracker — Cloudflare R2 checkpoints + D1 metrics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "boto3",
        "requests",
        "numpy",
    ],
    extras_require={
        "torch": ["torch"],
        "jax":   ["jax", "jaxlib"],
    },
    python_requires=">=3.9",
)
