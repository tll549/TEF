import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()
long_description = "See https://github.com/tll549/TEF for more details"

setuptools.setup(
    name="TEF",
    version="0.5.0",
    author="tll549",
    author_email="el@tll.tl",
    description="Ethan (tll549)'s Exploratory Functions",
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