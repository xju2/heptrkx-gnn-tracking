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
        "graph_nets",
        'tensorflow-gpu==1.14.0',
        'gast==0.2.2',
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
    setup_requires=['trackml'],
    classifiers=[
        "Programming Language :: Python :: 3.6",
    ],
    scripts=[
        'scripts/acts_seeding_eff',
        'scripts/create_seed_inputs',
        'scripts/duplicated_hits',
        'scripts/evaluate_gnn_models',
        'scripts/evaluate_event',
        'scripts/evaluate_doublet_NN',
        'scripts/fit_hits',
        'scripts/get_total_segments',
        'scripts/hits_graph_to_tuple',
        'scripts/make_doublets_from_NNs',
        'scripts/make_doublets_from_cuts',
        'scripts/make_graph',
        'scripts/make_pairs_for_training_segments',
        'scripts/make_trkx',
        'scripts/make_true_pairs_for_training_segments_mpi',
        'scripts/make_true_pairs',
        'scripts/merge_true_pairs',
        'scripts/merge_true_fake_pairs',
        'scripts/pairs_to_nx_graph',
        'scripts/peek_models',
        'scripts/plot_graph',
        'scripts/prepare_hitsgraph',
        'scripts/seeding_eff_purity_comparison',
        'scripts/segment_eff_purity',
        'scripts/select_pairs',
        'scripts/tf_train_pairs',
        'scripts/tf_train_pairs_all',
        'scripts/train_nx_graph',
        'scripts/track_eff_purity',
        'scripts/train_infomax',
        'scripts/view_training_log',
    ],
)
