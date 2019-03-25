"""
GNN models
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .model          import SegmentClassifier as mm
from .model_more     import SegmentClassifier as mm_more
from .model_smart    import SegmentClassifier as mm_sm


def get_model(model_name=None):

    # model_name could be used for future testing different models

    if model_name == "MORE":
        return mm_more()
    elif model_name == "SMART":
        return mm_sm()
    else:
        return mm()

