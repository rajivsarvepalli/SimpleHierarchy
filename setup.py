import setuptools
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simple-hierarchy-pytorch", 
    version="0.0.1",
    author="Rajiv Sarvepalli",
    include_package_data=True,
    author_email="rajiv@sarvepalli.net",
    description="simple hierarchal models in pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rajivsarvepalli/SimpleHierarchy",
    install_requires=[            # I get to this in a second
        'torch>=1.0',
    ],
    packages=setuptools.find_packages(exclude=("tests",)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    license="MIT"
)