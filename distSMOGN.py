from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from collections import Counter

from smogn import smoter
from smogn.phi import phi
from smogn.phi_ctrl_pts import phi_ctrl_pts
from smogn.over_sampling import over_sampling


# Most of the code is copied from the smogn package

def distSMOGN(data, y, thresh=0.8, num_partitions=4, pert=0.02, k_neigh=5):
    n = len(data)
    d = len(data.columns)

    # store original data types
    feat_dtypes_orig = [None] * d

    for j in range(d):
        feat_dtypes_orig[j] = data.iloc[:, j].dtype

    # determine column position for response variable y
    y_col = data.columns.get_loc(y)

    # move response variable y to last column
    if y_col < d - 1:
        cols = list(range(d))
        cols[y_col], cols[d - 1] = cols[d - 1], cols[y_col]
        data = data[data.columns[cols]]

    # store original feature headers and
    # encode feature headers to index position
    feat_names = list(data.columns)
    data.columns = range(d)

    # sort response variable y by ascending order
    y = pd.DataFrame(data[d - 1])
    y_sort = y.sort_values(by=d - 1)
    y_sort = y_sort[d - 1]

    phi_params = phi_ctrl_pts(y=y_sort)
    y_phi = phi(y=y_sort, ctrl_pts=phi_params)

    # determine bin (rare or normal) by bump classification
    bumps = [0]

    for i in range(0, len(y_sort) - 1):
        if (y_phi[i] >= thresh > y_phi[i + 1]) or (y_phi[i] < thresh <= y_phi[i + 1]):
            bumps.append(i + 1)

    bumps.append(n)

    # number of bump classes
    n_bumps = len(bumps) - 1

    # determine indicies for each bump classification
    b_index = {}

    for i in range(n_bumps):
        b_index.update({i: y_sort[bumps[i]:bumps[i + 1]]})

    # calculate over / under sampling percentage according to
    # bump class and user specified method ("balance" or "extreme")
    b = round(n / n_bumps)
    s_perc = []

    for i in b_index:
        s_perc.append(b / len(b_index[i]))

    # conduct over / under sampling and store modified training set
    data_new = pd.DataFrame()

    bins_r = pd.DataFrame()
    clustered_perc = {}
    for i in range(n_bumps):

        # no sampling
        if s_perc[i] == 1:
            # simply return no sampling
            # results to modified training set
            data_new = pd.concat([data.iloc[b_index[i].index], data_new])

        # under-sampling
        if s_perc[i] < 1:
            # drop observations in training set
            # considered 'normal' (not 'rare')
            omit_index = np.random.choice(
                a=list(b_index[i].index),
                size=int(s_perc[i] * len(b_index[i])),
                replace=False
            )

            omit_obs = data.drop(
                index=omit_index,
                axis=0
            )

            # concatenate under-sampling
            # results to modified training set
            data_new = pd.concat([omit_obs, data_new])

        # over-sampling
        if s_perc[i] > 1:
            # create dataset of under-represented entries
            bins_r = pd.concat([data.iloc[b_index[i].index], bins_r])
            for index in b_index[i].index:
                clustered_perc[index] = s_perc[i]

    kmeans = KMeans(n_clusters=num_partitions)
    kmeans.fit(bins_r)
    partitions = kmeans.predict(bins_r)
    clustered_data = {i: bins_r[partitions == i] for i in range(num_partitions)}

    # for cluster_idx, data_subset in clustered_data.items():
    #     data_subset = data_subset.drop(data_subset.columns[0], axis=1)
    #
    #     values = [clustered_perc[index] for index in data_subset.index]
    #     value_counts = Counter(values)
    #     most_common_value = value_counts.most_common(1)[0][0]
    #
    #     synth_obs = over_sampling(
    #         data=data_subset,
    #         index=list(data_subset.index),
    #         perc=most_common_value,
    #         pert=pert,
    #         k=k_neigh
    #     )
    #     # concatenate over-sampling
    #     # results to modified training set
    #     data_new = pd.concat([synth_obs, data_new])

    # rename feature headers to originals
    data_new.columns = feat_names

    # restore response variable y to original position
    if y_col < d - 1:
        cols = list(range(d))
        cols[y_col], cols[d - 1] = cols[d - 1], cols[y_col]
        data_new = data_new[data_new.columns[cols]]

    # restore original data types
    for j in range(d):
        data_new.iloc[:, j] = data_new.iloc[:, j].astype(feat_dtypes_orig[j])

    # return modified training set
    return data_new
