from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from setuptools import find_packages
from setuptools import setup

description="Use Graph Network to reconstruct tracks"

setup(
    name="heptrkx",
    version="0.0.1",
    description="Library for building tracks with Graph Nural Networks.",
    long_description=description,
    author="HEPTrkx",
    license="Apache License, Version 2.0",
    keywords=["graph networks", "track formation", "tracking", "machine learning"],
    url="https://github.com/xju2/hep-gnn-tracking",
    packages=find_packages(),
    install_requires=[
        "graph_nets==1.0.2",
        'tensorflow==1.12.0',
        "future",
        "networkx",
        "numpy",
        "scipy",
        "pandas",
        "setuptools",
        "six",
        "matplotlib",
        "sklearn",
        "torch==1.0.1",
        "torchvision==0.2.1",
        'sklearn',
        'pyyaml',
        'trackml==3'
    ],
    setup_requires=['trackml'],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    dependency_links=['https://github.com/LAL/trackml-library/tarball/master#egg=trackml-3'],
    scripts=[
        'bin/make_true_pairs_for_training_segments_mpi',
        'bin/merge_true_pairs',
        'bin/make_pairs_for_training_segments',
        'bin/select_pairs',
        'bin/tf_train_pairs',
        'bin/tf_train_pairs_all',
        'bin/train_nx_graph',
        'bin/evaluate_gnn_models',
        'bin/hits_graph_to_tuple',
        'bin/make_graphs_from_pair_NNs',
        'bin/pairs_to_nx_graph',
        'bin/get_total_segments',
    ],
)
