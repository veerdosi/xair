"""
Setup script for the XAIR package.
"""

from setuptools import setup, find_packages

setup(
    name="xair",
    version="0.1.0",
    description="Explainable AI Reasoning for LLMs",
    author="Veer Dosi",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.31.0",
        "scipy>=1.9.0", 
        "numpy>=1.24.0",
        "networkx>=3.0",
        "bitsandbytes>=0.40.0",
        "accelerate>=0.20.0",
    ],
    python_requires=">=3.8",
)