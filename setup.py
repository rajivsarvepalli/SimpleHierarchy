import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hierarchal-classification-pytorch-rajivsarvepalli", #
    version="0.0.1",
    author="Rajiv Sarvepalli",
    author_email="rajiv@sarvepalli.net",
    description="simple hierarchal model in pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rajivsarvepalli/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)