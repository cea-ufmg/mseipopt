#!/usr/bin/env python3

import setuptools

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()


CLASSIFIERS = """\
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved
License :: OSI Approved :: MIT License
Operating System :: MacOS
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Topic :: Software Development"""

setuptools.setup(
    name="mseipopt",
    version="0.1.dev1",
    author="Dimas Abreu Archanjo Dutra",
    author_email="dimasad@ufmg.br",
    description="The most simple ever python IPOPT interface.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/cea-ufmg/mseipopt",
    packages=setuptools.find_packages(),
    classifiers=CLASSIFIERS.split('\n'),
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    license="MIT",
    tests_require=["pytest"],
)
