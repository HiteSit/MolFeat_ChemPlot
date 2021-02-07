from setuptools import setup, find_packages

setup(
    name="ChemPlot", 
    version="",
    author="",
    author_email="",
    description="A python library for chemical space visualization.",
    long_description=long_description,
    long_description_content_type="",
    url="https://github.com/mcsorkun/ChemPlot",
    license="MIT"
    packages=find_packages(exclude=['docs']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
)