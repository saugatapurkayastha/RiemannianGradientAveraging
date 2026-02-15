import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rgrad_avg",
    version="0.1.0",
    author="Saugata Purkayastha and Sukannya Purkayastha",
    description="Riemannian Gradient Averaging Optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rgrad_avg",
    py_modules=["RGD"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
    ],
    python_requires=">=3.6",
)