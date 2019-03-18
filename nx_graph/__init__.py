"""
GNN models
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .model          import SegmentClassifier as mm

from .model_less     import SegmentClassifier as mm_less
from .model_more     import SegmentClassifier as mm_more
from .model_noLayerNorm import SegmentClassifier as mm_nonorm
from .model_noIntermediate import SegmentClassifier as mm_noint


def get_model(model_name=None):

    # model_name could be used for future testing different models

    if model_name == "LESS":
        return mm_less()
    elif model_name == "MORE":
        return mm_more()
    elif model_name == "NOLAYERNORM":
        print("Use model", model_name)
        return mm_nonorm()
    elif model_name == "NOINT":
        return mm_noint()
    else:
        pass


    return mm()
