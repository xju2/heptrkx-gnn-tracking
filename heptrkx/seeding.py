"""
Utils to evaluate seeding
"""
import glob
import pandas as pd
import numpy as np
import pickle

def read_triplets(seed_candidates):
    """
    Read the input seed candidates
    """
    if "pickle" in seed_candidates:
        if "*" in seed_candidates:
            all_files = glob.glob(seed_candidates)
            new_data = []
            for file_name in all_files:
                with open(file_name, 'rb') as f:
                    data = pickle.load(f)
                    for dd in data:
                        new_data.append((dd[0], dd[1], dd[2], dd[3]))
            df_seed = pd.DataFrame(new_data, columns=['evtid', 'h1', 'h2', 'h3'], dtype=np.int64)
        else:
            with open(seed_candidates, 'rb') as f:
                data = pickle.load(f)
                new_data = []
                for dd in data:
                    new_data.append((dd[0], dd[1], dd[2], dd[3]))
                    # idx = int(dd[0][10:])
                    # new_data.append((idx, dd[1], dd[2], dd[3]))
                df_seed = pd.DataFrame(new_data, columns=['evtid', 'h1', 'h2', 'h3'], dtype=np.int64)
    else:
        column_names = ['evtid', 'h1', 'h2', 'h3']
        if "*" in seed_candidates:
            all_files = glob.glob(seed_candidates)
            new_data = []
            for file_name in all_files:
                df_seed_tmp = pd.read_csv(file_name, header=None, names=column_names,)
                new_data.append(df_seed_tmp)
            df_seed = pd.concat(new_data)
        else:
            df_seed = pd.read_csv(seed_candidates, header=None,
                                names=column_names)
    return df_seed