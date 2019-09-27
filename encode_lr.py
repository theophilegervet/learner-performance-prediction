import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from scipy import sparse
import argparse
import os

from utils.queue import TimeWindowQueue


def phi(x):
    return np.log(1 + x)


def df_to_sparse(df, Q_mat, active_features, time_windows):
    """Build sparse dataset from dense dataset and q-matrix.

    Arguments:
        df (pandas DataFrame): output by prepare_data.py
        Q_mat (sparse array): q-matrix, output by prepare_data.py
        active_features (list of str): features
        time_windows (bool): if True, encode past wins/attempts with time windows to
            preserve temporal information

    Output:
        sparse_df (sparse array): sparse dataset where first 4 columns are the same as in df
    """
    num_items, num_skills = Q_mat.shape
    features = {}

    # Window lengths and counters
    window_lengths = [3600 * 24 * 30, 3600 * 24 * 7, 3600 * 24, 3600]
    num_windows = len(window_lengths) + 1
    counters = defaultdict(lambda: TimeWindowQueue(window_lengths))

    # Transform q-matrix into dictionary for fast lookup
    Q_mat_dict = {i: set() for i in range(num_items)}
    for i, j in np.argwhere(Q_mat == 1):
        Q_mat_dict[i].add(j)

    # Keep track of original dataset
    features['df'] = np.empty((0, len(df.keys())))

    # Skill features
    if 'skills' in active_features:
        features["skills"] = sparse.csr_matrix(np.empty((0, num_skills)))

    # Past attempts and wins features
    for key in ['attempts', 'wins']:
        if key in active_features:
            if time_windows:
                features[key] = sparse.csr_matrix(np.empty((0, (num_skills + 2) * num_windows)))
            else:
                features[key] = sparse.csr_matrix(np.empty((0, num_skills + 2)))

    # Build feature rows by iterating over users
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id][["user_id", "item_id", "timestamp", "correct"]].copy()
        df_user = df_user.values
        num_items_user = df_user.shape[0]

        skills = Q_mat[df_user[:, 1].astype(int)].copy()

        features['df'] = np.vstack((features['df'], df_user))

        item_ids = df_user[:, 1].reshape(-1, 1)
        labels = df_user[:, 3].reshape(-1, 1)

        if 'skills' in active_features:
            features['skills'] = sparse.vstack([features["skills"], sparse.csr_matrix(skills)])

        if 'attempts' in active_features:
            if time_windows:
                attempts = np.zeros((num_items_user, (num_skills + 2) * num_windows))

                for i, (item_id, ts) in enumerate(df_user[:, 1:3]):
                    # Past attempts for relevant skills
                    for skill_id in Q_mat_dict[item_id]:
                        counts = phi(np.array(counters[user_id, skill_id, "skill"].get_counters(ts)))
                        attempts[i, skill_id * num_windows:(skill_id + 1) * num_windows] = counts
                        counters[user_id, skill_id, "skill"].push(ts)

                    # Past attempts for item
                    counts = phi(np.array(counters[user_id, item_id, "item"].get_counters(ts)))
                    attempts[i, -2 * num_windows:-1 * num_windows] = counts
                    counters[user_id, item_id, "item"].push(ts)

                    # Past attempts for all items
                    counts = phi(np.array(counters[user_id].get_counters(ts)))
                    attempts[i, -1 * num_windows:] = counts
                    counters[user_id].push(ts)

            else:
                attempts = np.zeros((num_items_user, num_skills + 2))

                # Past attempts for relevant skills
                tmp = np.vstack((np.zeros(num_skills), skills))[:-1]
                attempts[:, :num_skills] = phi(np.cumsum(tmp, 0) * skills)

                # Past attempts for item
                onehot = OneHotEncoder(n_values=df_user[:, 1].max() + 1)
                item_ids_onehot = onehot.fit_transform(item_ids).toarray()
                tmp = np.vstack((np.zeros(item_ids_onehot.shape[1]), np.cumsum(item_ids_onehot, 0)))[:-1]
                attempts[:, -2] = phi(tmp[np.arange(num_items_user), df_user[:, 1]])

                # Past attempts for all items
                attempts[:, -1] = phi(np.arange(num_items_user))

            features['attempts'] = sparse.vstack([features['attempts'], sparse.csr_matrix(attempts)])

        if "wins" in active_features:
            if time_windows:
                wins = np.zeros((num_items_user, (num_skills + 2) * num_windows))

                for i, (item_id, ts, correct) in enumerate(df_user[:, 1:4]):
                    # Past wins for relevant skills
                    for skill_id in Q_mat_dict[item_id]:
                        counts = phi(np.array(counters[user_id, skill_id, "skill", "correct"].get_counters(ts)))
                        wins[i, skill_id * num_windows:(skill_id + 1) * num_windows] = counts
                        if correct:
                            counters[user_id, skill_id, "skill", "correct"].push(ts)

                    # Past wins for item
                    counts = phi(np.array(counters[user_id, item_id, "item", "correct"].get_counters(ts)))
                    wins[i, -2 * num_windows:-1 * num_windows] = counts
                    if correct:
                        counters[user_id, item_id, "item", "correct"].push(ts)

                    # Past wins for all items
                    counts = phi(np.array(counters[user_id, "correct"].get_counters(ts)))
                    wins[i, -1 * num_windows:] = counts
                    if correct:
                        counters[user_id, "correct"].push(ts)

            else:
                wins = np.zeros((num_items_user, num_skills + 2))

                # Past wins for relevant skills
                tmp = np.vstack((np.zeros(num_skills), skills))[:-1]
                corrects = np.hstack((np.array(0), df_user[:, 3])).reshape(-1, 1)[:-1]
                wins[:, :num_skills] = phi(np.cumsum(tmp * corrects, 0) * skills)

                # Past wins for item
                onehot = OneHotEncoder(n_values=df_user[:, 1].max() + 1)
                item_ids_onehot = onehot.fit_transform(item_ids).toarray()
                tmp = np.vstack((np.zeros(item_ids_onehot.shape[1]), np.cumsum(item_ids_onehot * labels, 0)))[:-1]
                wins[:, -2] = phi(tmp[np.arange(num_items_user), df_user[:, 1]])

                # Past wins for all items
                wins[:, -1] = phi(np.concatenate((np.zeros(1), np.cumsum(df_user[:, 3])[:-1])))

            features['wins'] = sparse.vstack([features['wins'], sparse.csr_matrix(wins)])

    # User and item one hot encodings
    onehot = OneHotEncoder()
    if 'users' in active_features:
        features['users'] = onehot.fit_transform(features["df"][:, 0].reshape(-1, 1))
    if 'items' in active_features:
        features['items'] = onehot.fit_transform(features["df"][:, 1].reshape(-1, 1))

    X = sparse.hstack([sparse.csr_matrix(features['df']),
                       sparse.hstack([features[x] for x in active_features])]).tocsr()
    return X


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode sparse feature matrix for logistic regression.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--users', action='store_true')
    parser.add_argument('--items', action='store_true')
    parser.add_argument('--skills', action='store_true')
    parser.add_argument('--wins', action='store_true')
    parser.add_argument('--attempts', action='store_true')
    parser.add_argument('--time_windows', action='store_true')
    args = parser.parse_args()

    data_path = os.path.join('data', args.dataset)

    all_features = ['users', 'items', 'skills', 'wins', 'attempts']
    active_features = [features for features in all_features if vars(args)[features]]
    features_suffix = ''.join([features[0] for features in active_features])
    if args.time_windows:
        features_suffix += '_tw'

    df = pd.read_csv(os.path.join(data_path, 'preprocessed_data.csv'), sep="\t")
    qmat = sparse.load_npz(os.path.join(data_path, 'q_mat.npz')).toarray()
    features = df_to_sparse(df, qmat, active_features, args.time_windows)
    sparse.save_npz(os.path.join(data_path, f"X-lr-{features_suffix}"), features)