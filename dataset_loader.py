import pandas as pd
import os


def get_csv_filepaths(folder_path: str):
    csv_filepaths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
    csv_filenames = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    return csv_filepaths, csv_filenames


def read_csv(file_path: str):
    data = pd.read_csv(file_path)
    return data


def load_csvs(folder_path: str):
    filepaths, filenames = get_csv_filepaths(folder_path)
    csvs = []

    for filepath in filepaths:
        csvs.append(read_csv(filepath))

    return filenames, csvs


