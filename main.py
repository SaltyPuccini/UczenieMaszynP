import time

import numpy as np
import pandas as pd
import smogn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from LDS import prepare_weights
from data_balance import create_balanced_datasets

from dataset_loader import load_csvs
from distSMOGN import distSMOGN
from models_creator import initialize_regression_models, test_regression_models


def main():
    balance_names = ["None", "SMOGN", "DistSMOGN", "LDS", "SMOGN_LDS", "DistSMOGN_LDS"]
    DATASET_DIR: str = "datasets"
    files, data_list = load_csvs(DATASET_DIR)
    regression_models = initialize_regression_models()

    n_splits = 2
    n_repeats = 1
    random_state = 0

    smogn_config = {'threshold': 0.5, 'pert': 0.023, 'knn': 5, 'rel_coef': 1.3, 'undersampling': False}
    dist_smogn_config = {'threshold': 0.5, 'pert': 0.023, 'knn': 5, 'num_partitions': 2, 'rel_coef': 1.3,
                         'undersampling': False}

    mae = np.zeros((len(files), len(regression_models), 6))
    rmse = np.zeros((len(files), len(regression_models), 6))

    for i, dataframe in enumerate(data_list):
        curr_dataframe = dataframe.copy()

        print(f"Training on {files[i]}...")
        text_columns = curr_dataframe.select_dtypes(include=['object']).columns.tolist()

        if text_columns:
            pd_y = curr_dataframe.iloc[:, -1]
            curr_dataframe = curr_dataframe.drop(columns=curr_dataframe.columns[-1])
            curr_dataframe = pd.get_dummies(curr_dataframe, columns=text_columns, prefix='', prefix_sep='')
            curr_dataframe = pd.concat([curr_dataframe, pd_y], axis=1)

        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        for fold_idx, (train_index, test_index) in enumerate(
                rkf.split(curr_dataframe.iloc[:, :-1], curr_dataframe.iloc[:, -1])):
            print(f"Balance data for fold {fold_idx}...")

            X_train, X_test = curr_dataframe.iloc[train_index, :-1], curr_dataframe.iloc[test_index, :-1]
            y_train, y_test = curr_dataframe.iloc[train_index, -1], curr_dataframe.iloc[test_index, -1]

            y_test.columns = [curr_dataframe.columns[-1]]
            y_train.columns = [curr_dataframe.columns[-1]]

            train_df = pd.concat((X_train, y_train), axis=1)

            balanced_dict = {"None": train_df.copy(),
                             "SMOGN": smogn.smoter(train_df.copy(), curr_dataframe.columns[-1], k=smogn_config['knn'],
                                                   pert=smogn_config['pert'],
                                                   rel_thres=smogn_config['threshold'],
                                                   under_samp=smogn_config['undersampling'],
                                                   rel_coef=smogn_config['rel_coef']),
                             "DistSMOGN": distSMOGN(train_df.copy(), curr_dataframe.columns[-1],
                                                    thresh=dist_smogn_config['threshold'],
                                                    num_partitions=dist_smogn_config['num_partitions'],
                                                    pert=dist_smogn_config['pert'], k_neigh=dist_smogn_config['knn'],
                                                    under_sampling=dist_smogn_config['undersampling'],
                                                    rel_coef=dist_smogn_config['rel_coef']),
                             }

            balanced_dict["LDS"] = balanced_dict["None"].copy()
            balanced_dict["SMOGN_LDS"] = balanced_dict["SMOGN"].copy()
            balanced_dict["DistSMOGN_LDS"] = balanced_dict["DistSMOGN"].copy()

            weights_dict = {
                "LDS": prepare_weights(train_df.copy()),
                "SMOGN_LDS": prepare_weights(balanced_dict["SMOGN"].copy()),
                "DistSMOGN_LDS": prepare_weights(balanced_dict["DistSMOGN"].copy()),
            }

            for model_idx, model in enumerate(regression_models):
                for balance_idx, (balance_type_name, data) in enumerate(balanced_dict.items()):
                    print(f"Training {model} on {balance_type_name}...")
                    current_model = clone(model)

                    if balance_type_name.__contains__("LDS") and type(model) is not MLPRegressor:
                        weights = weights_dict[balance_type_name].copy()
                        current_model.fit(data.iloc[:, :-1], data.iloc[:, -1], sample_weight=weights)

                    else:
                        current_model.fit(data.iloc[:, :-1], data.iloc[:, -1])

                    y_pred = current_model.predict(X_test)
                    a = y_test
                    print(mean_absolute_error(y_test, y_pred))
                    print(root_mean_squared_error(y_test, y_pred))
                    mae[i][model_idx][balance_idx] += mean_absolute_error(y_test, y_pred)
                    rmse[i][model_idx][balance_idx] += root_mean_squared_error(y_test, y_pred)

    mae = mae / n_splits / n_repeats
    rmse = rmse / n_splits / n_repeats

    for dataset in range(mae.shape[0]):
        for model in range(mae.shape[1]):
            for i in range(mae.shape[2]):
                with open('last_test.txt', 'a') as file:
                    file.write(f'{files[dataset]} --- {regression_models[model]} --- {balance_names[i]}\n')
                    file.write(f"Mean MAE: {mae[dataset][model][i]}\n")
                    file.write(f"Mean RMSE: {rmse[dataset][model][i]}\n")


if __name__ == '__main__':
    main()
