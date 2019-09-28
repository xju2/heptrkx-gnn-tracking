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
        "graph_nets@ https://github.com/deepmind/graph_nets/tarball/master",
        'tensorflow-gpu',
        "future",
        "networkx",
        "numpy",
        "scipy",
        "pandas",
        "setuptools",
        "six",
        "matplotlib",
        "torch",
        "torchvision",
        'sklearn',
        'pyyaml>=5.1',
        'trackml@ https://github.com/LAL/trackml-library/tarball/master#egg=trackml-3',
        'tables',
        'h5py',
    ],
    setup_requires=['trackml', 'graph_nets'],
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
        'scripts/duplicated_hits',
        'scripts/edge_cuts_efficiency',
        'scripts/edge_cuts_track_eff',
        'scripts/create_seed_inputs',
        'scripts/eval_doublet_NN',
        'scripts/prepare_hitsgraph',
        'scripts/merge_true_fake_pairs',
        'scripts/seeding_eff_purity_comparison',
        'scripts/fit_hits',
    ],
)
