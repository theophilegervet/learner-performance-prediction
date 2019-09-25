import os
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder


def phi(x):
    return np.log(1 + x)


def accumulate(x):
    return np.vstack((np.zeros(x.shape[1]), np.cumsum(x, 0)))[:-1]


def df_to_sparse(df, Q_mat, active_features, num_prev_interactions):
    """Build sparse dataset from dense dataset and q-matrix.

    Arguments:
        df (pandas DataFrame): output by prepare_data.py
        Q_mat (sparse array): q-matrix, output by prepare_data.py
        active_features (list of str): features
        num_prev_interactions (int): number of previous interactions to encode

    Output:
        sparse_df (sparse array): sparse dataset where first 4 columns are the same as in df
    """
    num_items, num_skills = Q_mat.shape
    onehot_items = OneHotEncoder(num_items)
    features = None

    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id][["user_id", "item_id", "timestamp", "correct"]].copy()
        df_user = df_user.values
        num_items_user = df_user.shape[0]

        attempts = np.zeros((num_items_user, num_items + num_skills + 1))
        wins = np.zeros((num_items_user, num_items + num_skills + 1))

        item_ids = df_user[:, 1].reshape(-1, 1)
        labels = df_user[:, 3].reshape(-1, 1)

        item_ids_onehot = onehot_items.fit_transform(item_ids).toarray()
        skills_onehot = Q_mat[item_ids.flatten()]

        # Previous interactions (item + correctness encoding)
        if num_prev_interactions > 0:
            prev_interactions = np.zeros((num_items_user, num_prev_interactions))
            encodings = (item_ids + labels * num_items).flatten()
            for i in range(num_prev_interactions):
                prev_interactions[i+1:, i] = encodings[:-i-1]

        # Past attempts for each item and total
        counts = accumulate(item_ids_onehot)
        attempts[:, :num_items] = phi(counts)
        attempts[:, -1] = phi(counts.sum(axis=1))

        # Past attempts for each skill
        attempts[:, num_items:num_items + num_skills] = phi(accumulate(skills_onehot))

        # Past wins for each item and total
        counts = accumulate(item_ids_onehot * labels)
        wins[:, :num_items] = phi(counts)
        wins[:, -1] = phi(counts.sum(axis=1))

        # Past wins for each skill
        wins[:, num_items:num_items + num_skills] = phi(accumulate(skills_onehot * labels))

        user_features = df_user
        if num_prev_interactions > 0:
            user_features = np.hstack((user_features, prev_interactions))
        if 'total' in active_features:
            user_features = np.hstack((user_features, attempts[:, -1:], wins[:, -1:]))
        if 'items' in active_features:
            user_features = np.hstack((user_features, attempts[:, :num_items], wins[:, :num_items]))
        if 'skills' in active_features:
            user_features = np.hstack((user_features,
                                       attempts[:, num_items:num_items + num_skills],
                                       wins[:, num_items:num_items + num_skills]))
        user_features = sparse.csr_matrix(user_features)
        features = user_features if features is None else sparse.vstack((features, user_features))

    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode sparse feature matrix for feedforward network baseline.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--total', action='store_true', help='If True, add total past attempts/wins.')
    parser.add_argument('--skills', action='store_true', help='If True, add past attempts/wins per skill.')
    parser.add_argument('--items', action='store_true', help='If True, add past attempts/wins per item.')
    parser.add_argument('--num_prev_interactions', type=int, default=1,
                        help='Number of previous interactions to include.')
    args = parser.parse_args()

    data_path = os.path.join('data', args.dataset)

    all_features = ['total', 'skills', 'items']
    active_features = [f for f in all_features if vars(args)[f]]
    features_suffix = ''.join([f[0] for f in active_features])

    df = pd.read_csv(os.path.join(data_path, 'preprocessed_data.csv'), sep="\t")
    Q_mat = sparse.load_npz(os.path.join(data_path, 'q_mat.npz')).toarray()
    X = df_to_sparse(df, Q_mat, active_features, args.num_prev_interactions)
    sparse.save_npz(os.path.join(data_path, f"X-ffw-{features_suffix}-{args.num_prev_interactions}"), X)