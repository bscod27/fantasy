import numpy as np
import pandas as pd
import random
import sys

from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


# set seeds
seed = 123
np.random.seed(seed)
random.seed(seed)


# unpack commands line args
assert len(sys.argv[1:]) == 1, "ERROR: Expected one command-line argument, specifying the model type [enet, knn, svr, rf, xgb]"
mod = sys.argv[1]


# load and sort data
traindat = pd.read_csv('../data/traindat.csv')
traindat.sort_values(by=['year', 'player'], inplace=True)


# define groups and split data into train/test
groups = traindat[traindat['year'].isin(range(2004, 2020))]['year']
train = traindat[traindat['year'].isin(range(2004, 2020))].drop(['player', 'team', 'year', 'posTE'], axis=1)
test = traindat[traindat['year'].isin(range(2020, 2024))].drop(['player', 'team', 'year', 'posTE'], axis=1)


# partition train and test into X/y
X_train, y_train = train.drop('ppg', axis=1), train['ppg']
X_test, y_test = test.drop('ppg', axis=1), test['ppg']


# set up grid search
if mod == 'enet':
    model = ElasticNet()
    grid = {
        'alpha': [0.001, 0.01, 0.1, 1, 10],
        'l1_ratio': [0.1, 0.5, 0.9]
    }

if mod == 'knn':
    model = KNeighborsRegressor()
    grid = {
        'n_neighbors': [3, 5, 7, 10, 15],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
    }

if mod == 'svr':
    model = SVR()
    grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2],
        'degree': [2, 3, 4]  # Only relevant for 'poly' kernel
    }

if mod == 'rf':
    model = RandomForestRegressor(n_jobs=10)
    grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

if mod == 'xgb':
    model = XGBRegressor(n_jobs=10)
    grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'subsample': [0.6, 0.8, 1.0],
        'n_estimators': [50, 100, 200],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }


# define the training mechanism
group_kfold = GroupKFold(n_splits=5)
grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=group_kfold, scoring='neg_mean_squared_error')


# search over parameter space to find best params, refit best model to entire train data
grid_search.fit(X_train, y_train, groups=groups)
final_model = grid_search.best_estimator_
final_model.fit(X_train, y_train)

# generate predictions on train and test sets and calculate metrics
preds_train = final_model.predict(X_train)
actuals_train = y_train

preds_test = final_model.predict(X_test)
actuals_test = y_test

train_rmse = np.sqrt(mean_squared_error(preds_train, actuals_train))
train_corr = pearsonr(preds_train, actuals_train)[0]

test_rmse = np.sqrt(mean_squared_error(preds_test, actuals_test))
test_corr = pearsonr(preds_test, actuals_test)[0]

print(
    f'''
    Train metrics
    \t RMSE: {train_rmse}
    \t Pearson: {train_corr}

    Test metrics
    \t RMSE: {test_rmse}
    \t Pearson: {test_corr}
    '''
)


# write to folder
pd.DataFrame({mod +'_train': preds_train}).to_csv('../preds/'+mod+'_train.csv', index=False)
pd.DataFrame({mod +'_test': preds_test}).to_csv('../preds/'+mod+'_test.csv', index=False)
stats = pd.DataFrame({'train': [train_rmse, train_corr], 'test': [train_rmse, train_corr]}, index=['rmse', 'corr'])
stats.to_csv('../preds/'+mod+'_stats.csv', index=True)
