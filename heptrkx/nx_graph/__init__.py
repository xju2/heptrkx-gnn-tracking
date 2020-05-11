"""
GNN models
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from heptrkx.nx_graph.model import SegmentClassifier as mm
from heptrkx.nx_graph.model_vary2 import SegmentClassifier as mm2
from heptrkx.nx_graph.model_vary3 import SegmentClassifier as mm3

def get_model(model_name=None):
    if model_name == 'vary2':
        return mm2()
    if model_name == "vary3":
        return mm3()
    return mm()