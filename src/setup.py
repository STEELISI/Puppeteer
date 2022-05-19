#!/usr/bin/env python

from distutils.core import setup
from os.path import abspath, dirname, join

from setuptools import find_packages

with open(
    join(dirname(abspath(__file__)), "puppeteer", "version.py")
) as version_file:
    exec(compile(version_file.read(), "version.py", "exec"))

setup(
    name="puppeteer",
    version=version,  # noqa
    author="Genevieve Bartlett",
    author_email="bartlett@isi.edu",
    description="modular dialog bot",
    url="https://github.com/STEELISI/Puppeteer",
    packages=find_packages(),
    # 3.6 and up, but not Python 4
    python_requires="~=3.6",
    install_requires=[],
    scripts=[],
    classifiers=["Programming Language :: Python :: 3"],
)
