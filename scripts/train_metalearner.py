import os
import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


# unpack files and read in meta learner X data
path = '../ensemble/'
train_files = [file for file in os.listdir(path) if 'train_' in file]
test_files = [file for file in os.listdir(path) if 'test_' in file]
model_files = [file for file in os.listdir(path) if 'models_' in file]

def read_data(files):
    df = pd.DataFrame()
    for i in range(len(files)):
        df = pd.concat([df,pd.read_csv(path + files[i])], axis=1)
    return df

X_train = read_data(train_files)
X_test = read_data(test_files)


# read in the y and group data
traindat = pd.read_csv('../data/traindat.csv')
traindat.sort_values(by=['year', 'player'], inplace=True)
group_idx = traindat[traindat['year'].isin(range(2004, 2020))].reset_index()['year']
y_train = traindat[traindat['year'].isin(range(2004, 2020))].reset_index()['ppg']
y_test = traindat[traindat['year'].isin(range(2020, 2024))].reset_index()['ppg']


# view the performance of the base models
for i in X_train.columns:
    print(i)
    print(f'Train RMSE: {np.sqrt(mean_squared_error(X_train[i], y_train))}')
    print(f'Train Pearson r: {pearsonr(X_train[i], y_train)[0]}')
    print(f'Test RMSE: {np.sqrt(mean_squared_error(X_test[i], y_test))}')
    print(f'Test Pearson r: {pearsonr(X_test[i], y_test)[0]}')
    print('\n')


# view the distributions of features to ensure they are on the same scale
print(pd.DataFrame([X_train.mean(axis=0), X_train.std(axis=0)], index=['mean', 'std']))


# train the meta learner
model = ElasticNet(random_state=123)
grid = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 1, 10],
    'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
}

k = 5
group_kfold = GroupKFold(n_splits=k)
grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=group_kfold, scoring='neg_mean_squared_error', verbose=0)
grid_search.fit(X_train, y_train, groups=group_idx)


# refit best model on all available data and save down
best_params = grid_search.best_params_
meta_learner = ElasticNet(**best_params)
meta_learner.fit(X_train, y_train)
yhat_test = meta_learner.predict(X_test)
yhat_train = meta_learner.predict(X_train)

print('meta learner')
print(f'Train RMSE: {np.sqrt(mean_squared_error(yhat_train, y_train))}')
print(f'Train Pearson r: {pearsonr(yhat_train, y_train)[0]}')
print(f'Test RMSE: {np.sqrt(mean_squared_error(yhat_test, y_test))}')
print(f'Test Pearson r: {pearsonr(yhat_train, y_train)[0]}')

with open('../ensemble/meta_learner.pkl', 'wb') as file:
    pickle.dump(meta_learner, file)
