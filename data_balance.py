import os.path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_balanced_datasets(dataset_name, balanced_dict, weights_dict, fold):
    for method in list(balanced_dict.keys()):

        if method.__contains__("LDS"):
            X = balanced_dict[method].iloc[:, -1].copy()
            weights = weights_dict[method].copy()

            plt.hist(X, weights=weights, bins=20, alpha=0.5, color='red', label='Weighted distribution')
            plt.hist(X, bins=20, alpha=0.5, color='green', label='Original distribution')
            plt.title('Histogram z nakładaniem wag')
            plt.xlabel('Wartość')
            plt.ylabel('Skumulowana waga')
        else:
            sns.kdeplot(balanced_dict["None"].iloc[:, -1], label="Original")
            sns.kdeplot(balanced_dict[method].iloc[:, -1], label="Modified")

        plt.legend()
        plt.title(f'{dataset_name} with {method} on fold {fold}')
        plt.savefig(os.path.join("plots", dataset_name + "_" + method + "_" + str(fold) + "_plot.png"))
        plt.close()
