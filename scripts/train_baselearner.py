import numpy as np
import pandas as pd
import random
import pickle
import sys

from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
        'model__alpha': [0.001, 0.01, 0.1, 1, 10],
        'model__l1_ratio': [0.1, 0.5, 0.9]
    }

if mod == 'knn':
    model = KNeighborsRegressor()
    grid = {
        'model__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
    }

if mod == 'svr':
    model = SVR()
    grid = {
        'model__kernel': ['rbf'],
        'model__C': [0.1, 1, 10, 100],
        'model__epsilon': [0.01, 0.1, 0.2],
    }

if mod == 'rf':
    model = RandomForestRegressor(n_jobs=10)
    grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20, 30],
        'model__max_features': ['auto', 'sqrt', 'log2']
    }

if mod == 'xgb':
    model = XGBRegressor(n_jobs=10)
    grid = {
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7, 10],
        'model__n_estimators': [100, 200, 300],
        'model__reg_alpha': [0, 0.1, 1], # l1 regularizaton
    }


# set up the normalization pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', model)
])


# define the training mechanism
k = 5
group_kfold = GroupKFold(n_splits=k)
grid_search = GridSearchCV(estimator=pipeline, param_grid=grid, cv=group_kfold, scoring='neg_mean_squared_error', verbose=3)


# iterate through folds, train models, stack/avg predictions to form meta train and test sets
fold = 0
meta_train = np.zeros(X_train.shape[0])
meta_test = np.zeros((X_test.shape[0], k))
meta_models = {}

for train_idx, val_idx in group_kfold.split(X_train, groups=groups):
    print(f'Fold: {str(fold)}')
    # chop up data
    X, y = X_train.iloc[train_idx], y_train.iloc[train_idx]
    group_idx = groups.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    # search the grid and find best fit
    grid_search.fit(X, y, groups = group_idx)
    best_fit = grid_search.best_estimator_
    # generate predictions
    meta_train[val_idx] = best_fit.predict(X_val)
    meta_test[:,fold] = best_fit.predict(X_test)
    meta_models[fold] = best_fit
    fold += 1


# compile and write results
out_train = pd.DataFrame({mod: meta_train})
out_test = pd.DataFrame({mod: meta_test.mean(axis=1)})

out_train.to_csv('../ensemble/train_base_'+mod+'.csv', index=False)
out_test.to_csv('../ensemble/test_base_'+mod+'.csv', index=False)

with open('../ensemble/models_base_'+mod+'.pkl', 'wb') as file:
    pickle.dump(meta_models, file)
