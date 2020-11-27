import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SimpleHierarchy", 
    version="0.0.1",
    author="Rajiv Sarvepalli",
    packages = ['SimpleHierarchy'],   # Chose the same as "name"

    author_email="rajiv@sarvepalli.net",
    description="simple hierarchal model in pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rajivsarvepalli/SimpleHierarchy",
    install_requires=[            # I get to this in a second
        'torch>=1.6',
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)