import torch
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


def get_preds(preds, item_ids, skill_ids, labels):
    preds = preds[labels >= 0]

    if (item_ids is not None):
        item_ids = item_ids[labels >= 0]
        preds = preds[torch.arange(preds.size(0)), item_ids]
    elif (skill_ids is not None):
        skill_ids = skill_ids[labels >= 0]
        preds = preds[torch.arange(preds.size(0)), skill_ids]

    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Write logistic regression and DKT predictions to a dataframe.')
    parser.add_argument('--X_file', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dkt_path', type=str)
    parser.add_argument('--max_iter', type=int, default=1000)
    args = parser.parse_args()

    dkt = torch.load(args.dkt_path)

    # Load sparse dataset
    X = csr_matrix(load_npz(args.X_file))

    user_ids = X[:, 0].toarray().flatten()
    users_train = pd.read_csv(f'data/{args.dataset}/preprocessed_data_train.csv', sep="\t")["user_id"].unique()
    users_test = pd.read_csv(f'data/{args.dataset}/preprocessed_data_test.csv', sep="\t")["user_id"].unique()

    X_train = X[np.where(np.isin(user_ids, users_train))]
    X_test = X[np.where(np.isin(user_ids, users_test))]
    y_train = X_train[:, 3].toarray().flatten()
    y_test = X_test[:, 3].toarray().flatten()

    # Train logistic regression
    model = LogisticRegression(solver="lbfgs", max_iter=args.max_iter)
    model.fit(X_train[:, 5:], y_train)
    lr_preds = model.predict_proba(X_test[:, 5:])[:, 1]

    lr_data = np.concatenate((X_test[:, :5].toarray(), lr_preds.reshape(-1, 1)), axis=1)
    lr_df = pd.DataFrame(data=lr_data, columns=["user_id", "item_id", "timestamp", "correct", "skill_id", "LR"])

    full_data = None

    for _, u_df in lr_df.groupby("user_id"):
        item_ids = torch.tensor(u_df["item_id"].values, dtype=torch.long)
        skill_ids = torch.tensor(u_df["skill_id"].values, dtype=torch.long)
        labels = torch.tensor(u_df["correct"].values, dtype=torch.long)
        item_inputs = torch.cat((torch.zeros(1, dtype=torch.long), item_ids * 2 + labels + 1))[:-1]
        skill_inputs = torch.cat((torch.zeros(1, dtype=torch.long), skill_ids * 2 + labels + 1))[:-1]

        labels = labels.unsqueeze(0)
        item_inputs = item_inputs.unsqueeze(0).cuda() if dkt.item_in else None
        skill_inputs = skill_inputs.unsqueeze(0).cuda() if dkt.skill_in else None
        item_ids = item_ids.unsqueeze(0).cuda() if dkt.item_out else None
        skill_ids = skill_ids.unsqueeze(0).cuda() if dkt.skill_out else None

        with torch.no_grad():
            dkt_preds, _ = dkt(item_inputs, skill_inputs)
            dkt_preds = torch.sigmoid(dkt_preds).cpu()
        dkt_preds = get_preds(dkt_preds, item_ids, skill_ids, labels).squeeze(0)

        new_data = np.hstack((u_df.values, dkt_preds.numpy().reshape(-1, 1)))
        full_data = new_data if full_data is None else np.vstack((full_data, new_data))

    full_df = pd.DataFrame(data=full_data,
                           columns=["user_id", "item_id", "timestamp", "correct", "skill_id", "LR", "DKT"])

    lr_auc = roc_auc_score(full_df["correct"], full_df["LR"])
    dkt_auc = roc_auc_score(full_df["correct"], full_df["DKT"])
    print(f"{args.dataset}: lr_auc={lr_auc}, dkt_auc={dkt_auc}")

    full_df.to_csv(f'data/{args.dataset}/predictions_test.csv', sep="\t", index=False)
