from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from joblib import dump
from data_balance import create_balanced_datasets
import os

from models_creator import initialize_regression_models

DATASET_DIR: str = "datasets"

stats = {
    'model_idx': [],
    'data_name': [],
    'data_idx': [],
    'fold_idx': [],
    'r2': []
}


def test():
    balanced_dict = create_balanced_datasets(DATASET_DIR)


def main():
    balanced_dict = create_balanced_datasets(DATASET_DIR)
    regression_models = initialize_regression_models()

    n_splits = 5  # Liczba podziałów
    n_repeats = 3  # Liczba powtórzeń
    random_state = 42  # Ziarno losowości

    # Pętla po modelach
    for model_idx, model in enumerate(regression_models):
        print(f"Training {model}...")
        for data_name, data_list in balanced_dict.items():
            print(f"Training on {data_name}...")
            for data_idx, data in enumerate(data_list):
                rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
                for fold_idx, (train_index, test_index) in enumerate(rkf.split(data, data.iloc[:, -1])):
                    X_train, X_test = data.iloc[train_index], data.iloc[test_index]
                    y_train, y_test = data.iloc[:, -1].iloc[train_index], data.iloc[:, -1].iloc[test_index]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)

                    stats['model_idx'].append(model_idx)
                    stats['data_name'].append(data_name)
                    stats['data_idx'].append(data_idx)
                    stats['fold_idx'].append(fold_idx)
                    stats['r2'].append(r2)

                    dump(model,
                         os.path.join("models", f'model_{model_idx}_{data_name}_{data_idx}_fold_{fold_idx}.joblib'))


if __name__ == '__main__':
    test()
