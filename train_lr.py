from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from scipy.sparse import load_npz, csr_matrix
import argparse
import numpy as np
import os


def compute_metrics(y_pred, y):
    acc = accuracy_score(y, np.round(y_pred))
    auc = roc_auc_score(y, y_pred)
    nll = log_loss(y, y_pred)
    return acc, auc, nll


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train logistic regression on feature matrix.')
    parser.add_argument('X_file', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--iter', type=int, default=400)
    args = parser.parse_args()
    
    data_path = os.path.join('data', args.dataset)
    features_suffix = (args.X_file.split("-")[1]).split(".")[0]

    # Load sparse dataset and q-matrix
    X = csr_matrix(load_npz(args.X_file))
    Q_mat = load_npz(os.path.join(data_path, "q_mat.npz"))
    
    # Student-level train-test split
    user_column = X[:, 0].toarray().flatten()
    users = np.unique(user_column)
    np.random.shuffle(users)
    split = int(0.8 * len(users))
    users_train, users_test = users[:split], users[split:]
    
    train = X[np.where(np.isin(user_column, users_train))]
    test = X[np.where(np.isin(user_column, users_test))]
    
    # First 4 columns are the original dataset including correct in column 3
    X_train, y_train = train[:, 4:], train[:, 3].toarray().flatten()
    X_test, y_test = test[:, 4:], test[:, 3].toarray().flatten()
    
    model = LogisticRegression(solver="lbfgs", max_iter=args.iter)
    model.fit(X_train, y_train)
    
    # Compute metrics
    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]
    acc_train, auc_train, nll_train = compute_metrics(y_pred_train, y_train)
    acc_test, auc_test, nll_test = compute_metrics(y_pred_test, y_test)
    print(f"{args.dataset}, {features_suffix}, train auc = {auc_train}, test auc = {auc_test}")
