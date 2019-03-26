from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from setuptools import find_packages
from setuptools import setup

description="Use Graph Network to reconstruct tracks"

setup(
    name="gnn_track",
    version="0.0.1",
    description="Library for building tracks with Graph Nural Networks.",
    long_description=description,
    author="HEPTrkx",
    license="Apache License, Version 2.0",
    keywords=["graph networks", "track formation", "tracking", "machine learning"],
    url="https://github.com/xju2/hep-gnn-tracking",
    packages=find_packages(),
    install_requires=[
        "graph_nets",
        "tensorflow==1.12",
        "future",
        "networkx",
        "numpy",
        "scipy",
        "pandas",
        "setuptools",
        "six",
        "dm-sonnet==1.23",
        "matplotlib",
        "sklearn",
        "torch",
        "torchvision"
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
