import argparse
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import GridSearchCV


def compute_metrics(y_pred, y):
    acc = accuracy_score(y, np.round(y_pred))
    auc = roc_auc_score(y, y_pred)
    nll = log_loss(y, y_pred)
    mse = brier_score_loss(y, y_pred)
    return acc, auc, nll, mse

def hyperParameterTuning(model, X_train, y_train):
    param_tuning = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators' : [100, 200, 500],
        'objective': ['reg:squarederror']
    }

    gsearch = GridSearchCV(estimator = model,
                           param_grid = param_tuning,                        
                           #scoring = 'neg_mean_absolute_error', #MAE
                           #scoring = 'neg_mean_squared_error',  #MSE
                           cv = 5,
                           n_jobs = -1,
                           verbose = 1)

    gsearch.fit(X_train, y_train)

    return gsearch.best_params_

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train logistic regression on sparse feature matrix.')
    parser.add_argument('--X_file', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--tune', type=int, default=0)

    args = parser.parse_args()

    features_suffix = (args.X_file.split("-")[-1]).split(".")[0]

    # Load sparse dataset
    X = csr_matrix(load_npz(args.X_file))

    train_df = pd.read_csv(f'data/{args.dataset}/preprocessed_data_train.csv', sep="\t")
    test_df = pd.read_csv(f'data/{args.dataset}/preprocessed_data_test.csv', sep="\t")
    
    # Student-wise train-test split
    user_ids = X[:, 0].toarray().flatten()
    users_train = train_df["user_id"].unique()
    users_test = test_df["user_id"].unique()
    train = X[np.where(np.isin(user_ids, users_train))]
    test = X[np.where(np.isin(user_ids, users_test))]

    # First 5 columns are the original dataset, including label in column 3
    X_train, y_train = train[:, 5:], train[:, 3].toarray().flatten()
    X_test, y_test = test[:, 5:], test[:, 3].toarray().flatten()

    # Train
    xgb_model = xgb.XGBClassifier(
        learning_rate = 0.01, 
        n_estimators = 200,
        verbose = 1,)
    if args.tune == 1:
        params = hyperParameterTuning(xgb_model, X_train, y_train)
        xgb_model.set_params(params)

    xgb_model.fit(X_train, y_train)

    y_pred_train = xgb_model.predict_proba(X_train)[:, 1]
    y_pred_test = xgb_model.predict_proba(X_test)[:, 1]

    # Write predictions to csv
    test_df[f"XG_{features_suffix}"] = y_pred_test
    test_df.to_csv(f'data/{args.dataset}/preprocessed_data_test.csv', sep="\t", index=False)

    acc_train, auc_train, nll_train, mse_train = compute_metrics(y_pred_train, y_train)
    acc_test, auc_test, nll_test, mse_test = compute_metrics(y_pred_test, y_test)
    print(f"{args.dataset}, features = {features_suffix}, "
          f"auc_train = {auc_train}, auc_test = {auc_test}, "
          f"mse_train = {mse_train}, mse_test = {mse_test}")