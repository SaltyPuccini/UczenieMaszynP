import os.path
import smogn
import pandas as pd
from LDS import prepare_weights
from dataset_loader import load_csvs
from distSMOGN import distSMOGN
import matplotlib.pyplot as plt
import seaborn as sns


def balance_dist_smogn(savedir, data: pd.DataFrame):
    # We don't want to modify the passed data
    copied_data: pd.DataFrame = data.copy()

    # Last column of the DF is the target value
    data_dist_smogn: pd.DataFrame = distSMOGN(data=copied_data, y=data.columns[-1])

    # Dataframe is saved to .csv file
    data_dist_smogn.to_csv(savedir + 'dist_smogn.csv', index=False)
    return data_dist_smogn


def balance_smogn(savedir, data: pd.DataFrame):
    # We don't want to modify the passed data
    copied_data: pd.DataFrame = data.copy()

    # Last column of the DF is the target value
    data_smogn: pd.DataFrame = smogn.smoter(data=copied_data, y=copied_data.columns[-1])

    # Dataframe is saved to .csv file
    data_smogn.to_csv(savedir + 'smogn.csv', index=False)
    return data_smogn


def balance_LDS(savedir, data: pd.DataFrame):
    # We don't want to modify the passed data
    copied_data: pd.DataFrame = data.copy()
    weights = prepare_weights(data)

    # Calculated weights are added as penultimate column of the DF
    copied_data.insert(len(copied_data.columns) - 1, 'W', weights)

    # Dataframe is saved to .csv file
    copied_data.to_csv(savedir + 'LDS.csv', index=False)
    return copied_data


def balance_smogn_LDS(savedir, smogn_data: pd.DataFrame):
    # We don't want to modify the passed data
    # This time we want to use LDS on already modified, smogn_data
    copied_data: pd.DataFrame = smogn_data.copy()
    weights = prepare_weights(smogn_data)

    # Calculated weights are added as penultimate column of the DF
    copied_data.insert(len(copied_data.columns) - 1, 'W', weights)

    # Dataframe is saved to .csv file
    copied_data.to_csv(savedir + 'smogn_LDS.csv', index=False)
    return copied_data


def plot_balanced_datasets(files, balanced_dict):
    for method in list(balanced_dict.keys()):
        for i in range(len(balanced_dict["NONE"])):
            result = balanced_dict[method][i].iloc[:, -1]

            if method.__contains__("LDS"):
                # questionable - should we really multiply it?
                result = [a * b for a, b in
                          zip(balanced_dict[method][i].iloc[:, -2], balanced_dict[method][i].iloc[:, -1])]

            sns.kdeplot(balanced_dict["NONE"][i].iloc[:, -1], label="Original")
            sns.kdeplot(result, label="Modified")
            plt.legend()
            dataset_name = files[i][:-4] + "_"
            plt.title(dataset_name + method)
            plt.savefig(os.path.join("plots", dataset_name + method + "_plot.png"))
            plt.close()


def create_balanced_datasets(dataset_dir: str):
    balanced_dict = {
        'NONE': [],
        'SMOGN': [],
        'LDS': [],
        'SMOGN + LDS': [],
        'DISTSMOGN': [],
        'DISTSMOGN + LDS': [],
    }

    filenames, dataframes = load_csvs(dataset_dir)

    for i, data in enumerate(dataframes):
        savedir = os.path.join("balanced_csv", filenames[i])[:-len('.csv')]
        balanced_dict['NONE'].append(data)

        if not os.path.exists(savedir + 'smogn.csv'):
            balanced_dict["SMOGN"].append(balance_smogn(savedir, data))
        else:
            balanced_dict["SMOGN"].append(pd.read_csv(savedir + 'smogn.csv'))

        if not os.path.exists(savedir + 'LDS.csv'):
            balanced_dict["LDS"].append(balance_LDS(savedir, data))
        else:
            balanced_dict["LDS"].append(pd.read_csv(savedir + 'LDS.csv'))

        if not os.path.exists(savedir + 'smogn_LDS.csv'):
            balanced_dict["SMOGN + LDS"].append(balance_smogn_LDS(savedir, balanced_dict["SMOGN"][i]))
        else:
            balanced_dict["SMOGN + LDS"].append(pd.read_csv(savedir + 'smogn_LDS.csv'))

        if not os.path.exists(savedir + 'dist_smogn.csv'):
            balanced_dict["DISTSMOGN"].append(balance_dist_smogn(savedir, data))
        else:
            balanced_dict["DISTSMOGN"].append(pd.read_csv(savedir + 'dist_smogn.csv'))

    plot_balanced_datasets(filenames, balanced_dict)

    return balanced_dict
