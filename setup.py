import os
from configparser import ConfigParser

from setuptools import find_packages, setup

config = ConfigParser()
rc = os.path.join(os.path.expanduser("~"), ".pypirc")
config.read(rc)

setup(
    name="code_base",
    version="0.0.1",
    description=("Codebase for any challenge"),
    author="Vladimir Sydorskyi",
    python_requires=">=3.8",
    packages=find_packages(),
)
