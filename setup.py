from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from setuptools import find_packages
from setuptools import setup

setup(
    name="gnn_track",
    version="1.0.4.dev",
    description="Library for building graph networks in Tensorflow and Sonnet.",
    long_description=description,
    author="DeepMind",
    license="Apache License, Version 2.0",
    keywords=["graph networks", "tensorflow", "sonnet", "machine learning"],
    url="https://github.com/deepmind/graph-nets",
    packages=find_packages(),
    install_requires=[
        "absl-py",
        "dm-sonnet==1.23",
        "future",
        "networkx",
        "numpy",
        "setuptools",
        "six",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
