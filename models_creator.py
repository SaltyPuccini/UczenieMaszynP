from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


def initialize_regression_models():
    svm_models = []
    cost_values = [10, 150, 300]
    gamma_values = [0.01, 0.001]

    for cost in cost_values:
        for gamma in gamma_values:
            svm_model = SVR(C=cost, gamma=gamma)
            svm_models.append(svm_model)

    rf_models = []
    mtry_values = [5, 7]
    ntree_values = [500, 750, 1500]

    for mtry in mtry_values:
        for ntree in ntree_values:
            rf_model = RandomForestRegressor(n_estimators=ntree, max_features=mtry)
            rf_models.append(rf_model)

    nn_models = []
    n_size = 10
    hidden_layers = [1, 5, 10]

    for l_size in hidden_layers:
        size = []
        for a in range(l_size):
            size.append(n_size)
        nn_model = MLPRegressor(size)
        nn_models.append(nn_model)

    return svm_models + rf_models + nn_models


def test_regression_models():
    svm_models = []
    cost_values = [300]
    gamma_values = [0.001]

    for cost in cost_values:
        for gamma in gamma_values:
            svm_model = SVR(C=cost, gamma=gamma)
            svm_models.append(svm_model)

    rf_models = []
    mtry_values = [5]
    ntree_values = [200]

    for mtry in mtry_values:
        for ntree in ntree_values:
            rf_model = RandomForestRegressor(n_estimators=ntree, max_features=mtry)
            rf_models.append(rf_model)

    return rf_models + svm_models
