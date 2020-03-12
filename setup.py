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
        "graph_nets",
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
        # 'h5py',
    ],
    extras_require={
        "tensorflow": ['tensorflow>=2.0'],
    },
    setup_requires=['trackml'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
    scripts=[
        # select doublets
        'scripts/create_hit_files',
        'scripts/create_predoublets',
        'scripts/train_doublets',
        'scripts/check_nnbased_doublet_sel',
        'scripts/merge_h5',
        'scripts/merge_predoublets',
        'scripts/select_doublet_nn',
        # evaluate triplets/seeds
        'scripts/create_seed_inputs',
        'scripts/seeding_perf',
        # doublet classifier
        # 'scripts/duplicated_hits',
        # 'scripts/evaluate_gnn_models',
        # 'scripts/evaluate_event',
        # 'scripts/evaluate_doublet_NN',
        # 'scripts/fit_hits',
        # 'scripts/get_total_segments',
        # 'scripts/hits_graph_to_tuple',
        # 'scripts/make_doublets_from_NNs',
        # 'scripts/make_doublets_from_cuts',
        # 'scripts/make_graph',
        # 'scripts/make_graph_hybrid',
        # 'scripts/make_pairs_for_training_segments',
        # 'scripts/make_trkx',
        # 'scripts/make_true_pairs',
        # 'scripts/merge_true_pairs',
        # 'scripts/merge_true_fake_pairs',
        # 'scripts/pairs_to_nx_graph',
        # 'scripts/peek_models',
        # 'scripts/plot_graph',
        # 'scripts/prepare_hitsgraph',
        # 'scripts/run_NN_segments',
        # 'scripts/seeding_eff_purity_comparison',
        # 'scripts/segment_eff_purity',
        # 'scripts/select_pairs',
        # 
        # 'scripts/train_nx_graph',
        # 'scripts/track_eff_purity',
        # 'scripts/train_infomax',
        # 'scripts/view_training_log',
        # 'scripts/make_doublets_from_NNs_fast',
        # 'scripts/evaluate_cutbased_doublets',
    ],
)
