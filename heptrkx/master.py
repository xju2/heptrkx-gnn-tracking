"""Master class
"""
import os

import numpy as np
import pandas as pd

from trackml.dataset import load_event

## predefined group info
vlids = [(7,2), (7,4), (7,6), (7,8), (7,10), (7,12), (7,14),
         (8,2), (8,4), (8,6), (8,8),
         (9,2), (9,4), (9,6), (9,8), (9,10), (9,12), (9,14),
         (12,2), (12,4), (12,6), (12,8), (12,10), (12,12),
         (13,2), (13,4), (13,6), (13,8),
         (14,2), (14,4), (14,6), (14,8), (14,10), (14,12),
         (16,2), (16,4), (16,6), (16,8), (16,10), (16,12),
         (17,2), (17,4),
         (18,2), (18,4), (18,6), (18,8), (18,10), (18,12)]
n_det_layers = len(vlids)

def Event(object):
    def __init__(self, evtdir):
        self._evt_dir = evtdir
        self._hits = None

    def read(self, evtid, merge_truth=True):
        prefix = os.path.join(os.path.expandvars(self.evt_dir),
                              'event{:09d}'.format(evtid))

        all_data = load_event(prefix,
                              parts=['hits', 'particles', 'truth', 'cells'])
        if all_data is None:
            return None

        hits, particles, truth, cells = all_data
        hits = hits.assign(evtid=evtid)

        ## add pT to particles
        px = particles.px
        py = particles.py
        pt = np.sqrt(px**2 + py**2)
        particles = particles.assign(pt=pt)
        self._evtid = evtid
        self._hits = hits
        self._particles = particles
        self._truth = truth
        self._cells = cells

        if merge_truth:
            self.merge_truth_info_to_hits()
        return (self._hits, self._particles, self._truth, self._cells)

    def merge_truth_info_to_hits(self, inplace=True):
        if not self._hits:
            return None

        hits = self._hits
        hits = hits.merge(truth, on='hit_id', how='left')
        hits = hits.merge(particles, on='particle_id', how='left')

        # noise hits does not have particle info
        # yielding NaN value
        hits = hits.fillna(value=0)

        # Assign convenient layer number [0-47]
        vlid_groups = hits.groupby(['volume_id', 'layer_id'])
        hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i)
                          for i in range(n_det_layers)])

        # add new features
        x = hits.x
        y = hits.y
        z = hits.z
        absz = np.abs(z)
        r = np.sqrt(x**2 + y**2) # distance from origin in transverse plane
        r3 = np.sqrt(r**2 + z**2) # in 3D
        phi = np.arctan2(hits.y, hits.x)
        theta = np.arccos(z/r3)
        eta = -np.log(np.tan(theta/2.))

        tpx = hits.tpx
        tpy = hits.tpy
        tpt = np.sqrt(tpx**2 + tpy**2)

        hits = hits.assign(r=r, phi=phi, eta=eta, r3=r3, absZ=absz, tpt=tpt)

        # add hit indexes to column hit_idx
        hits = hits.rename_axis('hit_idx').reset_index()
        if inplace:
            self._hits = hits
        return hits

    def filter_hits(self, layers, inplace=True):
        """keep hits that are in the layers"""
        barrel_hits = self._hits[self._hits.layer.isin(layers)]
        if inplace:
            self._hits = barrel_hits
        return barrel_hits

    def remove_noise_hits(self, inplace=True):
        no_noise = self._hits[self._hits.particle_id > 0]
        if inplace:
            self._hits = no_noise
        return no_noise
