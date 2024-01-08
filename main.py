import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error

from sklearn.model_selection import RepeatedKFold
from joblib import dump
from data_balance import create_balanced_datasets
import os

from dataset_loader import get_filenames
from models_creator import initialize_regression_models, test_regression_models

DATASET_DIR: str = "datasets"
files = get_filenames(DATASET_DIR)


def test():
    balanced_dict = create_balanced_datasets(DATASET_DIR)
    regression_models = test_regression_models()

    n_splits = 3  # Liczba podziałów
    n_repeats = 1  # Liczba powtórzeń
    random_state = 0  # Ziarno losowości

    for model_idx, model in enumerate(regression_models):
        print(f"Training {model}...")
        for data_name, data_list in balanced_dict.items():
            print(f"Training on {data_name}...")
            for data_idx, data in enumerate(data_list):
                print(f"Training on {files[data_idx]}...")
                ready_data = data
                weights = np.array([])
                text_columns = ready_data.select_dtypes(include=['object']).columns.tolist()

                if data_name.__contains__("LDS"):
                    weights = np.array(ready_data.iloc[:, -2])
                    ready_data = ready_data.drop(columns=ready_data.columns[-2])

                if text_columns:
                    pd_y = ready_data.iloc[:, -1]
                    ready_data = ready_data.drop(columns=ready_data.columns[-1])
                    ready_data = pd.get_dummies(ready_data, columns=text_columns, prefix='', prefix_sep='')
                    ready_data = pd.concat([ready_data, pd_y], axis=1)

                rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
                mae_list = []
                rmse_list = []
                for fold_idx, (train_index, test_index) in enumerate(rkf.split(ready_data, ready_data.iloc[:, -1])):
                    X_train, X_test = ready_data.iloc[train_index], ready_data.iloc[test_index]
                    y_train, y_test = ready_data.iloc[:, -1].iloc[train_index], ready_data.iloc[:, -1].iloc[test_index]
                    if np.any(weights):
                        train_weights = weights[train_index]
                        model.fit(X_train, y_train, train_weights)
                    else:
                        model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    mae_list.append(mean_absolute_error(y_test, y_pred))
                    rmse_list.append(root_mean_squared_error(y_test, y_pred))

                print(sum(mae_list) / len(mae_list))
                print(sum(rmse_list) / len(rmse_list))


def main():
    balanced_dict = create_balanced_datasets(DATASET_DIR)
    regression_models = initialize_regression_models()

    n_splits = 5
    n_repeats = 3
    random_state = 13

    for model_idx, model in enumerate(regression_models):
        print(f"Training {model}...")
        for data_name, data_list in balanced_dict.items():
            print(f"Training on {data_name}...")
            for data_idx, data in enumerate(data_list):
                print(f"Training on {files[data_idx]}...")
                ready_data = data
                weights = np.array([])
                text_columns = ready_data.select_dtypes(include=['object']).columns.tolist()

                if data_name.__contains__("LDS"):
                    weights = np.array(ready_data.iloc[:, -2])
                    ready_data = ready_data.drop(columns=ready_data.columns[-2])

                if text_columns:
                    pd_y = ready_data.iloc[:, -1]
                    ready_data = ready_data.drop(columns=ready_data.columns[-1])
                    ready_data = pd.get_dummies(ready_data, columns=text_columns, prefix='', prefix_sep='')
                    ready_data = pd.concat([ready_data, pd_y], axis=1)

                rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
                mae_list = []
                rmse_list = []
                for fold_idx, (train_index, test_index) in enumerate(rkf.split(ready_data, ready_data.iloc[:, -1])):
                    X_train, X_test = ready_data.iloc[train_index], ready_data.iloc[test_index]
                    y_train, y_test = ready_data.iloc[:, -1].iloc[train_index], ready_data.iloc[:, -1].iloc[test_index]
                    if np.any(weights):
                        train_weights = weights[train_index]
                        model.fit(X_train, y_train, train_weights)
                    else:
                        model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    mae_list.append(mean_absolute_error(y_test, y_pred))
                    rmse_list.append(root_mean_squared_error(y_test, y_pred))

                    dump(model,
                         os.path.join("models", f'model_{model_idx}_{data_name}_{data_idx}_fold_{fold_idx}.joblib'))

                print(sum(mae_list) / len(mae_list))
                print(sum(rmse_list) / len(rmse_list))


if __name__ == '__main__':
    test()
