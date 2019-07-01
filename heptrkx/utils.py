"""Handle read and write objects"""

import os
import glob
import re

import numpy as np
def evtids_at_disk(evt_dir):
    all_files = glob.glob(os.path.join(evt_dir, '*hits*'))
    evtids = np.sort([int(
        re.search('event([0-9]*)', os.path.basename(x).split('-')[0]).group(1))
        for x in all_files])
    return evtids
