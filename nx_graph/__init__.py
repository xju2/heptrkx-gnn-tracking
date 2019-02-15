"""
GNN models
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .model_mlp      import SegmentClassifier as mlp
from .model_mlp_tanh import SegmentClassifier as mlp_tanh
from .model          import SegmentClassifier as mm

def get_model(model_name):
    if model_name == 'model_mlp':
        return mlp()
    elif model_name == 'model_mlp_tanh':
        return mlp_tanh()
    else:
        return mm()
