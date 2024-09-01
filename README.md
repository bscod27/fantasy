# fantasy
Predict player points per game (half ppr) using stacked ensemble machine learning, leveraging Sports Reference player statistics from the previous 20 years. 

## Reproducibility
To reproduce this work, clone the repository, navigate to the root directory, and download/activate the environment: 
```
git clone https://github.com/bscod27/fantasy.git
cd fantasy
conda env create -f environment.yml
conda activate fantasy 
```
Results can be reproduced by executing `scripts/scraper.py` >> `scripts/wrangle.R` >> `scripts/train_baslearner.py` >> `scripts/train_metalearner.py` >> `scripts/generate_predictions.py`:

1. `scripts/scraper.py` - web scrapes Sports Reference player statistics from the last 20 years and writes it as `data/data_[t-20]_[t].csv`; requires a command-line argument specifying the current year, t
2. `scripts/wrangle.R` - engineers lag 1 features by player and partitions the data into `data/traindat.csv` (training data with labels) and `data/newdat.csv` (most recent year statistics to use as inputs for predictions)
3. `scripts/train_baselearner.py`- trains several lower-level base learner using 5-fold groupwise (by year) cross-validation, holding out the most recent 4 years as a test set; requires a command-line argument specifying a model type within [enet, knn, svr, rf, xgb]; writes the following files:
    - `ensemble/train_base_[model].csv` - transformed train set (stacked out-of-fold predictions)
    - `ensemble/test_base_[model].csv` - transformed test set (averaged across models trained in each fold)
    - `ensemble/models_base_[model].pkl` - pickled models trained in each fold 
4. `scripts/train_metalearner.py` - trains an Elastic Net model on the transformed feature space and prints the model performance; saves the pickled meta learner as `ensemble/meta_learner.pkl` 
5. `scripts/generate_predictions.py`- leverages the saved models to generate predictions on `data/newdat.csv`; writes these predictions as `preds/predictions.csv`