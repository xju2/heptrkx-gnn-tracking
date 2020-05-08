from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from setuptools import find_packages
from setuptools import setup

description="Use Graph Network to reconstruct tracks"

setup(
    name="heptrkx",
    version="1.0.0",
    description="Library for building tracks with Graph Nural Networks.",
    long_description=description,
    author="HEPTrkx",
    license="Apache License, Version 2.0",
    keywords=["graph networks", "track formation", "tracking", "machine learning"],
    url="https://github.com/xju2/hep-gnn-tracking",
    packages=find_packages(),
    install_requires=[
        "tensorflow >= 2.1",
        "graph_nets>=1.1",
        'gast',
        "future",
        "networkx",
        "numpy",
        "scipy",
        "pandas",
        "setuptools",
        "six",
        "matplotlib",
        'sklearn',
        'pyyaml>=5.1',
        'trackml@ https://github.com/LAL/trackml-library/tarball/master#egg=trackml-3',
        'tables',
        'more-itertools',
        # 'h5py',
    ],
    extras_require={
    },
    setup_requires=['trackml'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
    scripts=[
        'scripts/event_summary',
        # select doublets
        'scripts/create_hit_files',
        'scripts/create_predoublets',
        'scripts/create_predoublets_mpi',
        'scripts/train_doublets',
        'scripts/check_nnbased_doublet_sel',
        'scripts/merge_h5',
        'scripts/merge_predoublets',
        'scripts/select_doublet_nn',
        # evaluate triplets/seeds
        'scripts/create_seed_inputs',
        'scripts/seeding_perf',
        # event studies
        'scripts/duplicated_hits',
        # random
        'scripts/peek_models',
        'scripts/view_training_log',
        'scripts/trim_doublets',
        # GNN
        'scripts/train_nx_graph',
        'scripts/train_nx_graph_single',
        'scripts/train_nx_graph_distributed',
        'scripts/train_distributed',
        'scripts/train_nx_graph_tpu',
        # track candidates
        'scripts/tracks_from_triplet_graph',
        'scripts/create_tfrec_doublets',
    ],
)