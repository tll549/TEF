import setuptools

import re
import sys

with open('TEF/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            _, _, __version__ = line.replace("'", '').split()
            break

# with open("README.md", "r") as fh:
#     long_description = fh.read()
long_description = "See https://github.com/tll549/TEF for more details"

setuptools.setup(
    name="TEF",
    version=__version__,
    author="tll549",
    author_email="el@tll.tl",
    description="tll549's Exploratory Functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tll549/TEF",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)