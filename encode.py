import os
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils.queue import TimeWindowQueue


def phi(x):
    return np.log(1 + x)


WINDOW_LENGTHS = [3600 * 24 * 30, 3600 * 24 * 7, 3600 * 24, 3600]
NUM_WINDOWS = len(WINDOW_LENGTHS) + 1


def df_to_sparse(df, Q_mat, active_features):
    """Build sparse dataset from dense dataset and q-matrix.

    Arguments:
        df (pandas DataFrame): output by prepare_data.py
        Q_mat (sparse array): q-matrix, output by prepare_data.py
        active_features (list of str): features

    Output:
        sparse_df (sparse array): sparse dataset where first 5 columns are the same as in df
    """
    num_items, num_skills = Q_mat.shape
    features = {}

    # Counters for continuous time windows
    counters = defaultdict(lambda: TimeWindowQueue(WINDOW_LENGTHS))

    # Transform q-matrix into dictionary for fast lookup
    Q_mat_dict = {i: set() for i in range(num_items)}
    for i, j in np.argwhere(Q_mat == 1):
        Q_mat_dict[i].add(j)

    # Keep track of original dataset
    features['df'] = np.empty((0, len(df.keys())))

    # Skill features
    if 's' in active_features:
        features["s"] = sparse.csr_matrix(np.empty((0, num_skills)))

    # Past attempts and wins features
    for key in ['a', 'w']:
        if key in active_features:
            if 'tw' in active_features:
                features[key] = sparse.csr_matrix(np.empty((0, (num_skills + 2) * NUM_WINDOWS)))
            else:
                features[key] = sparse.csr_matrix(np.empty((0, num_skills + 2)))

    # Build feature rows by iterating over users
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id][["user_id", "item_id", "timestamp", "correct", "skill_id"]].copy()
        df_user = df_user.values
        num_items_user = df_user.shape[0]

        skills = Q_mat[df_user[:, 1].astype(int)].copy()

        features['df'] = np.vstack((features['df'], df_user))

        item_ids = df_user[:, 1].reshape(-1, 1)
        labels = df_user[:, 3].reshape(-1, 1)

        # Current skills one hot encoding
        if 's' in active_features:
            features['s'] = sparse.vstack([features["s"], sparse.csr_matrix(skills)])

        # Attempts
        if 'a' in active_features:
            # Time windows
            if 'tw' in active_features:
                attempts = np.zeros((num_items_user, (num_skills + 2) * NUM_WINDOWS))

                for i, (item_id, ts) in enumerate(df_user[:, 1:3]):
                    # Past attempts for relevant skills
                    if 'sc' in active_features:
                        for skill_id in Q_mat_dict[item_id]:
                            counts = phi(np.array(counters[user_id, skill_id, "skill"].get_counters(ts)))
                            attempts[i, skill_id * NUM_WINDOWS:(skill_id + 1) * NUM_WINDOWS] = counts
                            counters[user_id, skill_id, "skill"].push(ts)

                    # Past attempts for item
                    if 'ic' in active_features:
                        counts = phi(np.array(counters[user_id, item_id, "item"].get_counters(ts)))
                        attempts[i, -2 * NUM_WINDOWS:-1 * NUM_WINDOWS] = counts
                        counters[user_id, item_id, "item"].push(ts)

                    # Past attempts for all items
                    if 'tc' in active_features:
                        counts = phi(np.array(counters[user_id].get_counters(ts)))
                        attempts[i, -1 * NUM_WINDOWS:] = counts
                        counters[user_id].push(ts)

            # Counts
            else:
                attempts = np.zeros((num_items_user, num_skills + 2))

                # Past attempts for relevant skills
                if 'sc' in active_features:
                    tmp = np.vstack((np.zeros(num_skills), skills))[:-1]
                    attempts[:, :num_skills] = phi(np.cumsum(tmp, 0) * skills)

                # Past attempts for item
                if 'ic' in active_features:
                    onehot = OneHotEncoder(n_values=df_user[:, 1].max() + 1)
                    item_ids_onehot = onehot.fit_transform(item_ids).toarray()
                    tmp = np.vstack((np.zeros(item_ids_onehot.shape[1]), np.cumsum(item_ids_onehot, 0)))[:-1]
                    attempts[:, -2] = phi(tmp[np.arange(num_items_user), df_user[:, 1]])

                # Past attempts for all items
                if 'tc' in active_features:
                    attempts[:, -1] = phi(np.arange(num_items_user))

            features['a'] = sparse.vstack([features['a'], sparse.csr_matrix(attempts)])

        # Wins
        if "w" in active_features:
            # Time windows
            if 'tw' in active_features:
                wins = np.zeros((num_items_user, (num_skills + 2) * NUM_WINDOWS))

                for i, (item_id, ts, correct) in enumerate(df_user[:, 1:4]):
                    # Past wins for relevant skills
                    if 'sc' in active_features:
                        for skill_id in Q_mat_dict[item_id]:
                            counts = phi(np.array(counters[user_id, skill_id, "skill", "correct"].get_counters(ts)))
                            wins[i, skill_id * NUM_WINDOWS:(skill_id + 1) * NUM_WINDOWS] = counts
                            if correct:
                                counters[user_id, skill_id, "skill", "correct"].push(ts)

                    # Past wins for item
                    if 'ic' in active_features:
                        counts = phi(np.array(counters[user_id, item_id, "item", "correct"].get_counters(ts)))
                        wins[i, -2 * NUM_WINDOWS:-1 * NUM_WINDOWS] = counts
                        if correct:
                            counters[user_id, item_id, "item", "correct"].push(ts)

                    # Past wins for all items
                    if 'tc' in active_features:
                        counts = phi(np.array(counters[user_id, "correct"].get_counters(ts)))
                        wins[i, -1 * NUM_WINDOWS:] = counts
                        if correct:
                            counters[user_id, "correct"].push(ts)

            # Counts
            else:
                wins = np.zeros((num_items_user, num_skills + 2))

                # Past wins for relevant skills
                if 'sc' in active_features:
                    tmp = np.vstack((np.zeros(num_skills), skills))[:-1]
                    corrects = np.hstack((np.array(0), df_user[:, 3])).reshape(-1, 1)[:-1]
                    wins[:, :num_skills] = phi(np.cumsum(tmp * corrects, 0) * skills)

                # Past wins for item
                if 'ic' in active_features:
                    onehot = OneHotEncoder(n_values=df_user[:, 1].max() + 1)
                    item_ids_onehot = onehot.fit_transform(item_ids).toarray()
                    tmp = np.vstack((np.zeros(item_ids_onehot.shape[1]), np.cumsum(item_ids_onehot * labels, 0)))[:-1]
                    wins[:, -2] = phi(tmp[np.arange(num_items_user), df_user[:, 1]])

                # Past wins for all items
                if 'tc' in active_features:
                    wins[:, -1] = phi(np.concatenate((np.zeros(1), np.cumsum(df_user[:, 3])[:-1])))

            features['w'] = sparse.vstack([features['w'], sparse.csr_matrix(wins)])

    # User and item one hot encodings
    onehot = OneHotEncoder()
    if 'u' in active_features:
        features['u'] = onehot.fit_transform(features["df"][:, 0].reshape(-1, 1))
    if 'i' in active_features:
        features['i'] = onehot.fit_transform(features["df"][:, 1].reshape(-1, 1))

    X = sparse.hstack([sparse.csr_matrix(features['df']),
                       sparse.hstack([features[x] for x in features.keys() if x != 'df'])]).tocsr()
    return X


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode sparse feature matrix for logistic regression.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('-u', action='store_true',
                        help='If True, include user one hot encoding.')
    parser.add_argument('-i', action='store_true',
                        help='If True, include item one hot encoding.')
    parser.add_argument('-s', action='store_true',
                        help='If True, include skills many hot encoding .')
    parser.add_argument('-ic', action='store_true',
                        help='If True, include item historical counts.')
    parser.add_argument('-sc', action='store_true',
                        help='If True, include skills historical counts.')
    parser.add_argument('-tc', action='store_true',
                        help='If True, include total historical counts.')
    parser.add_argument('-w', action='store_true',
                        help='If True, historical counts include wins.')
    parser.add_argument('-a', action='store_true',
                        help='If True, historical counts include attempts.')
    parser.add_argument('-tw', action='store_true',
                        help='If True, historical counts are encoded as time windows.')
    args = parser.parse_args()

    data_path = os.path.join('data', args.dataset)
    df = pd.read_csv(os.path.join(data_path, 'preprocessed_data.csv'), sep="\t")
    df = df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]
    Q_mat = sparse.load_npz(os.path.join(data_path, 'q_mat.npz')).toarray()

    all_features = ['u', 'i', 's', 'ic', 'sc', 'tc', 'w', 'a', 'tw']
    active_features = [features for features in all_features if vars(args)[features]]
    features_suffix = ''.join(active_features)

    X = df_to_sparse(df, Q_mat, active_features)
    sparse.save_npz(os.path.join(data_path, f"X-{features_suffix}"), X)