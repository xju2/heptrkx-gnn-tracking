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
        'tensorflow-gpu==1.12.0',
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
        'pyyaml>=5.1',
        'trackml@ https://github.com/LAL/trackml-library/tarball/master#egg=trackml-3',
        'tables==3.4.4',
        'h5py==2.8.0',
    ],
    setup_requires=['trackml'],
    classifiers=[
        "Programming Language :: Python :: 3.6",
    ],
    scripts=[
        'scripts/make_true_pairs_for_training_segments_mpi',
        'scripts/merge_true_pairs',
        'scripts/make_pairs_for_training_segments',
        'scripts/select_pairs',
        'scripts/tf_train_pairs',
        'scripts/tf_train_pairs_all',
        'scripts/train_nx_graph',
        'scripts/evaluate_gnn_models',
        'scripts/hits_graph_to_tuple',
        'scripts/make_doublets_from_NNs',
        'scripts/make_doublets_from_cuts',
        'scripts/pairs_to_nx_graph',
        'scripts/get_total_segments',
        'scripts/make_graph',
        'scripts/plot_graph',
        'scripts/make_trkx',
    ],
)
