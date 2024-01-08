from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from collections import Counter
import random as rd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm

# load dependencies - internal
from smogn.box_plot_stats import box_plot_stats
from smogn.dist_metrics import euclidean_dist, heom_dist, overlap_dist
from smogn.phi import phi
from smogn.phi_ctrl_pts import phi_ctrl_pts
from smogn.over_sampling import over_sampling


# Most of the code is copied from the smogn package, I just needed to add minor modifications
# https://pypi.org/project/smogn/
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

    text_columns = bins_r.select_dtypes(include=['object']).columns.tolist()

    encoded_df = bins_r

    if text_columns:
        encoded_df = pd.get_dummies(bins_r, columns=text_columns, prefix='', prefix_sep='')
        encoded_df.columns = range(len(encoded_df.columns))

    kmeans = KMeans(n_clusters=num_partitions)
    kmeans.fit(encoded_df)
    partitions = kmeans.predict(encoded_df)

    clustered_data = {i: bins_r[partitions == i] for i in range(num_partitions)}

    for cluster_idx, data_subset in clustered_data.items():
        values = [clustered_perc[index] for index in data_subset.index]
        value_counts = Counter(values)
        most_common_value = value_counts.most_common(1)[0][0]

        synth_obs = distSMOGN_over_sampling(
            data=data_subset,
            index=list(data_subset.index),
            perc=most_common_value,
            pert=pert,
            k=k_neigh
        )
        # concatenate over-sampling
        # results to modified training set
        data_new = pd.concat([synth_obs, data_new])

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


def distSMOGN_over_sampling(

        # arguments / inputs
        data,  # training set
        index,  # index of input data
        perc,  # over / under sampling
        pert,  # perturbation / noise percentage
        k  # num of neighs for over-sampling

):
    # store dimensions of data subset
    n = len(data)
    d = len(data.columns)

    # store original data types
    feat_dtypes_orig = [None] * d

    for j in range(d):
        feat_dtypes_orig[j] = data.iloc[:, j].dtype

    # find non-negative numeric features
    feat_non_neg = []
    num_dtypes = ["int64", "float64"]

    for j in range(d):
        if data.iloc[:, j].dtype in num_dtypes and any(data.iloc[:, j] > 0):
            feat_non_neg.append(j)

    # find features without variation (constant features)
    feat_const = data.columns[data.nunique() == 1]

    # temporarily remove constant features
    if len(feat_const) > 0:

        # create copy of orignal data and omit constant features
        data_orig = data.copy()
        data = data.drop(data.columns[list(feat_const)], axis=1)

        # store list of features with variation
        feat_var = list(data.columns.values)

        # reindex features with variation
        for i in range(d - len(feat_const)):
            data.rename(columns={
                data.columns[i]: i
            }, inplace=True)

        # store new dimension of feature space
        d = len(data.columns)

    # create copy of data containing variation
    data_var = data.copy()

    # create global feature list by column index
    feat_list = list(data.columns.values)

    # create nominal feature list and
    # label encode nominal / categorical features
    # (strictly label encode, not one hot encode)
    feat_list_nom = []
    nom_dtypes = ["object", "bool", "datetime64"]

    for j in range(d):
        if data.dtypes[j] in nom_dtypes:
            feat_list_nom.append(j)
            data.iloc[:, j] = pd.Categorical(pd.factorize(
                data.iloc[:, j])[0])

    data = data.apply(pd.to_numeric)

    # create numeric feature list
    feat_list_num = list(set(feat_list) - set(feat_list_nom))

    # calculate ranges for numeric / continuous features
    # (includes label encoded features)
    feat_ranges = list(np.repeat(1, d))

    if len(feat_list_nom) > 0:
        for j in feat_list_num:
            feat_ranges[j] = max(data.iloc[:, j]) - min(data.iloc[:, j])
    else:
        for j in range(d):
            feat_ranges[j] = max(data.iloc[:, j]) - min(data.iloc[:, j])

    # subset feature ranges to include only numeric features
    # (excludes label encoded features)
    feat_ranges_num = [feat_ranges[i] for i in feat_list_num]

    # subset data by either numeric / continuous or nominal / categorical
    data_num = data.iloc[:, feat_list_num]
    data_nom = data.iloc[:, feat_list_nom]

    # get number of features for each data type
    feat_count_num = len(feat_list_num)
    feat_count_nom = len(feat_list_nom)

    # calculate distance between observations based on data types
    # store results over null distance matrix of n x n
    dist_matrix = np.ndarray(shape=(n, n))

    for i in tqdm(range(n), ascii=True, desc="dist_matrix"):
        for j in range(n):

            # utilize euclidean distance given that
            # data is all numeric / continuous
            if feat_count_nom == 0:
                dist_matrix[i][j] = euclidean_dist(
                    a=data_num.iloc[i],
                    b=data_num.iloc[j],
                    d=feat_count_num
                )

            # utilize heom distance given that
            # data contains both numeric / continuous
            # and nominal / categorical
            if feat_count_nom > 0 and feat_count_num > 0:
                dist_matrix[i][j] = heom_dist(

                    # numeric inputs
                    a_num=data_num.iloc[i],
                    b_num=data_num.iloc[j],
                    d_num=feat_count_num,
                    ranges_num=feat_ranges_num,

                    # nominal inputs
                    a_nom=data_nom.iloc[i],
                    b_nom=data_nom.iloc[j],
                    d_nom=feat_count_nom
                )

            # utilize hamming distance given that
            # data is all nominal / categorical
            if feat_count_num == 0:
                dist_matrix[i][j] = overlap_dist(
                    a=data_nom.iloc[i],
                    b=data_nom.iloc[j],
                    d=feat_count_nom
                )

    # determine indicies of k nearest neighbors
    # and convert knn index list to matrix
    knn_index = [None] * n

    for i in range(n):
        knn_index[i] = np.argsort(dist_matrix[i])[1:k + 1]

    knn_matrix = np.array(knn_index)

    # calculate max distances to determine if gaussian noise is applied
    # (half the median of the distances per observation)
    max_dist = [None] * n

    for i in range(n):
        max_dist[i] = box_plot_stats(dist_matrix[i])["stats"][2] / 2

    # number of new synthetic observations for each rare observation
    x_synth = int(perc - 1)

    # total number of new synthetic observations to generate
    n_synth = int(n * (perc - 1 - x_synth))

    # randomly index data by the number of new synthetic observations
    r_index = np.random.choice(
        a=tuple(range(0, n)),
        size=n_synth,
        replace=False,
        p=None
    )

    # create null matrix to store new synthetic observations
    synth_matrix = np.ndarray(shape=((x_synth * n + n_synth), d))

    if x_synth > 0:
        for i in tqdm(range(n), ascii=True, desc="synth_matrix"):

            # determine which cases are 'safe' to interpolate
            safe_list = np.where(
                dist_matrix[i, knn_matrix[i]] < max_dist[i])[0]

            for j in range(x_synth):

                # randomly select a k nearest neighbor
                neigh = int(np.random.choice(
                    a=tuple(range(k)),
                    size=1))

                # conduct synthetic minority over-sampling
                # technique for regression (smoter)
                if neigh in safe_list:
                    diffs = data.iloc[
                            knn_matrix[i, neigh], 0:(d - 1)] - data.iloc[
                                                               i, 0:(d - 1)]
                    synth_matrix[i * x_synth + j, 0:(d - 1)] = data.iloc[
                                                               i, 0:(d - 1)] + rd.random() * diffs

                    # randomly assign nominal / categorical features from
                    # observed cases and selected neighbors
                    for x in feat_list_nom:
                        synth_matrix[i * x_synth + j, x] = [data.iloc[
                                                                knn_matrix[i, neigh], x], data.iloc[
                                                                i, x]][round(rd.random())]

                    # generate synthetic y response variable by
                    # inverse distance weighted
                    for z in feat_list_num:
                        a = abs(data.iloc[i, z] - synth_matrix[
                            i * x_synth + j, z]) / feat_ranges[z]
                        b = abs(data.iloc[knn_matrix[
                            i, neigh], z] - synth_matrix[
                                    i * x_synth + j, z]) / feat_ranges[z]

                    if len(feat_list_nom) > 0:
                        a = a + sum(data.iloc[
                                        i, feat_list_nom] != synth_matrix[
                                        i * x_synth + j, feat_list_nom])
                        b = b + sum(data.iloc[knn_matrix[
                            i, neigh], feat_list_nom] != synth_matrix[
                                        i * x_synth + j, feat_list_nom])

                    if a == b:
                        synth_matrix[i * x_synth + j,
                        (d - 1)] = data.iloc[i, (d - 1)] + data.iloc[
                            knn_matrix[i, neigh], (d - 1)] / 2
                    else:
                        synth_matrix[i * x_synth + j,
                        (d - 1)] = (b * data.iloc[
                            i, (d - 1)] + a * data.iloc[
                                        knn_matrix[i, neigh], (d - 1)]) / (a + b)

                # conduct synthetic minority over-sampling technique
                # for regression with the introduction of gaussian
                # noise (smoter-gn)
                else:
                    if max_dist[i] > pert:
                        t_pert = pert
                    else:
                        t_pert = max_dist[i]

                    index_gaus = i * x_synth + j

                    for x in range(d):
                        if pd.isna(data.iloc[i, x]):
                            synth_matrix[index_gaus, x] = None
                        else:
                            synth_matrix[index_gaus, x] = data.iloc[
                                                              i, x] + float(np.random.normal(
                                loc=0,
                                scale=np.std(data.iloc[:, x]),
                                size=1) * t_pert)

                            if x in feat_list_nom:
                                if len(data.iloc[:, x].unique() == 1):
                                    synth_matrix[
                                        index_gaus, x] = data.iloc[0, x]
                                else:
                                    probs = [None] * len(
                                        data.iloc[:, x].unique())

                                    for z in range(len(
                                            data.iloc[:, x].unique())):
                                        probs[z] = len(
                                            np.where(data.iloc[
                                                     :, x] == data.iloc[:, x][z]))

                                    synth_matrix[index_gaus, x] = rd.choices(
                                        population=data.iloc[:, x].unique(),
                                        weights=probs,
                                        k=1)

    if n_synth > 0:
        count = 0

        for i in tqdm(r_index, ascii=True, desc="r_index"):

            # determine which cases are 'safe' to interpolate
            safe_list = np.where(
                dist_matrix[i, knn_matrix[i]] < max_dist[i])[0]

            # randomly select a k nearest neighbor
            neigh = int(np.random.choice(
                a=tuple(range(0, k)),
                size=1))

            # conduct synthetic minority over-sampling
            # technique for regression (smoter)
            if neigh in safe_list:
                diffs = data.iloc[
                        knn_matrix[i, neigh], 0:(d - 1)] - data.iloc[i, 0:(d - 1)]
                synth_matrix[x_synth * n + count, 0:(d - 1)] = data.iloc[
                                                               i, 0:(d - 1)] + rd.random() * diffs

                # randomly assign nominal / categorical features from
                # observed cases and selected neighbors
                for x in feat_list_nom:
                    synth_matrix[x_synth * n + count, x] = [data.iloc[
                                                                knn_matrix[i, neigh], x], data.iloc[
                                                                i, x]][round(rd.random())]

                # generate synthetic y response variable by
                # inverse distance weighted
                for z in feat_list_num:
                    a = abs(data.iloc[i, z] - synth_matrix[
                        x_synth * n + count, z]) / feat_ranges[z]
                    b = abs(data.iloc[knn_matrix[i, neigh], z] - synth_matrix[
                        x_synth * n + count, z]) / feat_ranges[z]

                if len(feat_list_nom) > 0:
                    a = a + sum(data.iloc[i, feat_list_nom] != synth_matrix[
                        x_synth * n + count, feat_list_nom])
                    b = b + sum(data.iloc[
                                    knn_matrix[i, neigh], feat_list_nom] != synth_matrix[
                                    x_synth * n + count, feat_list_nom])

                if a == b:
                    synth_matrix[x_synth * n + count, (d - 1)] = data.iloc[
                                                                     i, (d - 1)] + data.iloc[
                                                                     knn_matrix[i, neigh], (d - 1)] / 2
                else:
                    synth_matrix[x_synth * n + count, (d - 1)] = (b * data.iloc[
                        i, (d - 1)] + a * data.iloc[
                                                                      knn_matrix[i, neigh], (d - 1)]) / (a + b)

            # conduct synthetic minority over-sampling technique
            # for regression with the introduction of gaussian
            # noise (smoter-gn)
            else:
                if max_dist[i] > pert:
                    t_pert = pert
                else:
                    t_pert = max_dist[i]

                for x in range(d):
                    if pd.isna(data.iloc[i, x]):
                        synth_matrix[x_synth * n + count, x] = None
                    else:
                        synth_matrix[x_synth * n + count, x] = data.iloc[
                                                                   i, x] + float(np.random.normal(
                            loc=0,
                            scale=np.std(data.iloc[:, x]),
                            size=1) * t_pert)

                        if x in feat_list_nom:
                            if len(data.iloc[:, x].unique() == 1):
                                synth_matrix[
                                    x_synth * n + count, x] = data.iloc[0, x]
                            else:
                                probs = [None] * len(data.iloc[:, x].unique())

                                for z in range(len(data.iloc[:, x].unique())):
                                    probs[z] = len(np.where(
                                        data.iloc[:, x] == data.iloc[:, x][z])
                                    )

                                synth_matrix[
                                    x_synth * n + count, x] = rd.choices(
                                    population=data.iloc[:, x].unique(),
                                    weights=probs,
                                    k=1
                                )

            # close loop counter
            count = count + 1

    # convert synthetic matrix to dataframe
    data_new = pd.DataFrame(synth_matrix)

    # synthetic data quality check
    if sum(data_new.isnull().sum()) > 0:
        raise ValueError("oops! synthetic data contains missing values")

    # replace label encoded values with original values
    for j in feat_list_nom:
        code_list = data.iloc[:, j].unique()
        cat_list = data_var.iloc[:, j].unique()

        for x in code_list:
            data_new.iloc[:, j] = data_new.iloc[:, j].replace(x, cat_list[x])

    # reintroduce constant features previously removed
    if len(feat_const) > 0:
        data_new.columns = feat_var

        for j in range(len(feat_const)):
            data_new.insert(
                loc=int(feat_const[j]),
                column=feat_const[j],
                value=np.repeat(
                    data_orig.iloc[0, feat_const[j]],
                    len(synth_matrix))
            )

    # convert negative values to zero in non-negative features
    for j in feat_non_neg:
        # data_new.iloc[:, j][data_new.iloc[:, j] < 0] = 0
        data_new.iloc[:, j] = data_new.iloc[:, j].clip(lower=0)

    # return over-sampling results dataframe
    return data_new
