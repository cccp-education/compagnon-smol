from setuptools import setup, find_packages

setup(
    name="compagnon",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "pydantic-ai>=0.0.17",
    ],
)
