import os
import pickle
import regex as re
import numpy as np
import pandas as pd


# load base learners
path = '../ensemble/'
model_files = [file for file in os.listdir(path) if 'models_' in file]


# load meta learner
with open(path + 'meta_learner.pkl', 'rb') as file:
    meta_learner = pickle.load(file)


# load newdata
dat = pd.read_csv('../data/newdat.csv')
players = dat[['player', 'team', 'posQB', 'posWR', 'posRB', 'posTE']]
lagged_dep = dat['lag_ppg']
newdat = dat.drop(['player', 'team', 'year', 'posTE', 'lag_ppg'], axis=1)
newdat['lag_ppg'] = lagged_dep


# transform newdata to meta model__max_features
meta_features = {}

for i in model_files:
    name = re.findall('.*_(.*).pkl', i)[0]
    print(str(name)+'...')
    with open(path + i, 'rb') as file:
        base_models = pickle.load(file)

    preds = np.zeros((newdat.shape[0], 5)) # because there were 5-fols
    for j in range(len(base_models)):
        mod = base_models[j]
        preds[:,j] = mod.predict(newdat)

    meta_features[name] = preds.mean(axis=1) # avg across cols


# feed transformed features through meta learner to get predictions
out = pd.concat(
    [players,pd.DataFrame({'preds_ppg': meta_learner.predict(pd.DataFrame(meta_features))})],
    axis=1
).sort_values('preds_ppg', ascending=False)


# write output
out.to_csv('../preds/predictions.csv', index=False)
