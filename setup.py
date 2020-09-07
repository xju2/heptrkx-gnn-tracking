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
    author="Xiangyang Ju",
    author_email="xiangyang.ju@gmail.com",
    license="Apache License, Version 2.0",
    keywords=["graph networks", "track formation", "tracking", "machine learning"],
    url="https://github.com/xju2/hep-gnn-tracking",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "tensorflow >= 2.2",
        "graph_nets>=1.1",
        'gast',
        "future",
        "networkx>=2.4",
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
    package_data = {
        "heptrkx": ["config/*.yaml"]
    },
    extras_require={
    },
    setup_requires=['trackml'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
    scripts=[
        'scripts/event_summary',
        # preparation
        'scripts/create_tfrec_doublets',
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
        # GNN training
        'scripts/train_nx_graph',
        'scripts/train_nx_graph_single',
        'scripts/train_nx_graph_distributed',
        'scripts/train_distributed',
        'scripts/train_nx_graph_tpu',
        # Distributed training
        'scripts/hvd_distributed',
        # GNN evaluation
        'scripts/evaluate_edge_classifier',
        'scripts/evaluate_distributed_edge_classifier',
        # track candidates
        'scripts/tracks_from_triplet_graph',
        'scripts/tracks_from_doublet_graph',
        # utils
        'scripts/list_evtids',
        'scripts/view_checkpoints',
        # doublet studies
        'scripts/select_pairs',
        "scripts/eff_purity_comparison",
    ],
)
