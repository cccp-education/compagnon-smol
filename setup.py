from setuptools import setup, find_packages

setup(
    name="compagnon-smol",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "smolagents>=1.0.0",
        "pydantic>=2.9.2",
        "fastapi>=0.115.2",
        "pillow>=11.0.0",
        "packaging>=23.2",
        "pgvector>=0.3.5",
        "PyMonad>=2.4.0",
        "PyYAML~=6.0.1",
        "setuptools~=74.1.2",
        "huggingface-hub>=0.28.1",
        "assertpy>=1.1",
        "pytest>=8.3.4",
        "pytest-asyncio>=0.25.3",
    ],
)
