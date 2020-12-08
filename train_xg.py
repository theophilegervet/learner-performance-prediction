import argparse
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
from sklearn.grid_search import GridSearchCV


def compute_metrics(y_pred, y):
    acc = accuracy_score(y, np.round(y_pred))
    auc = roc_auc_score(y, y_pred)
    nll = log_loss(y, y_pred)
    mse = brier_score_loss(y, y_pred)
    return acc, auc, nll, mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train logistic regression on sparse feature matrix.')
    parser.add_argument('--X_file', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--iter', type=int, default=1000)
    args = parser.parse_args()

    features_suffix = (args.X_file.split("-")[-1]).split(".")[0]

    param_grid = {
        'max_depth':range(3,10,2),
        'min_child_weight':range(1,6,2),
        'subsample':[i/10.0 for i in range(6,10)],
        'colsample_bytree':[i/10.0 for i in range(6,10)],
        'gamma':[i/10.0 for i in range(0,5)],
        'eta':[.3, .2, .1, .05, .01, .005],
        'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
    }
    
    model = xgb.XGBClassifier(n_jobs=1, verbose=1)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2, cv=2, n_jobs=-1)
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
    # model = LogisticRegression(solver="lbfgs", max_iter=args.iter)
    model = grid_search
    model.fit(X_train, y_train)

    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]

    # Write predictions to csv
    test_df[f"XG_{features_suffix}"] = y_pred_test
    test_df.to_csv(f'data/{args.dataset}/preprocessed_data_test.csv', sep="\t", index=False)

    acc_train, auc_train, nll_train, mse_train = compute_metrics(y_pred_train, y_train)
    acc_test, auc_test, nll_test, mse_test = compute_metrics(y_pred_test, y_test)
    print(f"{args.dataset}, features = {features_suffix}, "
          f"auc_train = {auc_train}, auc_test = {auc_test}, "
          f"mse_train = {mse_train}, mse_test = {mse_test}")