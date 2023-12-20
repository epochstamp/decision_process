"""
Setup for 'decision_process' module.

"""
import os
import pathlib
from setuptools import setup, find_namespace_packages


# Package meta-data.
NAME = "uliege-decision_process"
PACKAGE = "uliege.decision_process"
DESCRIPTION = "Generic decision process for end-to-end control."
URL = "https://bitbucket.org/smartgrids_core/decision_process"
EMAIL = "saittahar@uliege.be"
AUTHOR = "University of Liege"
KEYWORDS = "decision_process control linear_programming"
REQUIRES_PYTHON = ">=3.7.4"


# What packages are required for this module to be executed?
REQUIRED = [
    "Pyomo>=5.7.*",
    "numpy>=1.20.*",
    "pydantic>=1.8.*",
    "sortedcontainers>=2.3.*",
    "gym>=0.18.*",
]

EXTRA_REQUIRED = {
    "tests": [
        "pytest==5.0.1",
        "pytest-cov",
        "hypothesis",
        "pytest-parallel",
        "pytest-progress",
    ],
    "dev": ["flake8", "black", "pre-commit"],
}


# What extra files are required for this module to be executed?
EXTRA_FILES = []


# Setup
setup(
    name=NAME,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    keywords=KEYWORDS,
    url=URL,
    packages=find_namespace_packages(include=[PACKAGE, f"{PACKAGE}.*"]),
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRED,
    extras_require=EXTRA_REQUIRED,
    package_data={PACKAGE: EXTRA_FILES},
    include_package_data=True,
    zip_safe=False,
    # Configure package versioning from source code management tool (i.e. git).
    use_scm_version={
        "local_scheme": lambda *_: "",  # do not prepend dirty-related tag to version
        "write_to": os.path.join("./", PACKAGE.replace(".", "/"), "_version.py"),
    },
    setup_requires=["setuptools_scm"],
    namespace_packages=["uliege"],
)
