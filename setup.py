from setuptools import find_packages, setup

import undertale

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="undertale",
    version=undertale.__version__,
    author=undertale.__author__,
    author_email="undertale@ll.mit.edu",
    url="https://llcad-github.llan.ll.mit.edu/undertale/undertale",
    description=undertale.__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    license_files=["LICENSE.txt"],
    packages=find_packages(),
    python_requires="==3.10.*",
    install_requires=[
        "polars",
        "scikit-learn",
        "datatrove",
        "datasets",
        "transformers",
        "torch",
        "lightning",
        "tensorboard",
        "openai",
        "capstone",
        "pyhidra",
        "r2pipe",
        "rzpipe",
        "pefile",
        "networkx",
        "bs4",
        "requests",
        "filelock",
        "lief",
        "unix_ar",
        "cpp_demangle",
        "itanium-demangler",
        "rust-demangler",
    ],
    extras_require={
        "development": [
            "isort",
            "black",
            "flake8",
            "pip-tools",
            "pre-commit",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
