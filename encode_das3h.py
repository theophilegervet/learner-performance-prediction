import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from scipy import sparse
import argparse
import os

from utils.queue import TimeWindowQueue


NUM_TIME_WINDOWS = 5


def df_to_sparse(df, Q_mat, active_features, tw=None):
    """Build sparse dataset from dense dataset and q-matrix.

    Arguments:
        df (pandas DataFrame): output by prepare_data.py
        Q_mat (sparse array): q-matrix, output by prepare_data.py
        active_features (list of str): features
        tw (str or None): "tw_tc" or "tw_items" encode time windows features, respectively one
            feature per (time window x skill) and one feature per time window (assume only
            attempts on the item are relevant), while None drops temporal information (like in PFA)
            while retaining per skill statistics

    Output:
        sparse_df (sparse array): sparse dataset where first 4 columns are the same as in df
    """
    num_items, num_skills = Q_mat.shape
    features = {}
    counters = defaultdict(lambda: TimeWindowQueue()) # Counters for time windows
    onehot = OneHotEncoder()

    # Transform q-matrix into dictionary for fast lookup
    Q_mat_dict = {i: set() for i in range(num_items)}
    for i, j in np.argwhere(Q_mat == 1):
        Q_mat_dict[i].add(j)

    # Keep track of original dataset
    features['df'] = np.empty((0, len(df.keys())))

    # Skill features
    if 'skills' in active_features:
        features["skills"] = sparse.csr_matrix(np.empty((0, num_skills)))

    # Attempt, win and fail features
    if 'attempts' in active_features:
        if tw == "tw_kc":
            features["attempts"] = sparse.csr_matrix(np.empty((0, num_skills * NUM_TIME_WINDOWS)))
        elif tw == "tw_items":
            features["attempts"] = sparse.csr_matrix(np.empty((0, NUM_TIME_WINDOWS)))
        else:
            features["attempts"] = sparse.csr_matrix(np.empty((0, num_skills)))

    if 'wins' in active_features:
        if tw == "tw_kc":
            features["wins"] = sparse.csr_matrix(np.empty((0, num_skills * NUM_TIME_WINDOWS)))
        elif tw == "tw_items":
            features["wins"] = sparse.csr_matrix(np.empty((0, NUM_TIME_WINDOWS)))
        else:
            features["wins"] = sparse.csr_matrix(np.empty((0, num_skills)))

    if 'fails' in active_features:
        features["fails"] = sparse.csr_matrix(np.empty((0, num_skills)))

    # Build feature rows iterating over users
    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id][["user_id", "item_id", "timestamp", "correct"]].copy()
        df_user.sort_values(by="timestamp", inplace=True)
        df_user = df_user.values
        num_items_user = df_user.shape[0]

        features['df'] = np.vstack((features['df'], df_user))

        if 'skills' in active_features:
            skills = Q_mat[df_user[:, 1].astype(int)].copy()
            features['skills'] = sparse.vstack([features["skills"], sparse.csr_matrix(skills)])

        if "attempts" in active_features:
            skills = Q_mat[df_user[:, 1].astype(int)].copy()
            if tw == "tw_kc":
                attempts = np.zeros((num_items_user, NUM_TIME_WINDOWS * num_skills))
                for i, (item_id, timestamp) in enumerate(df_user[:, 1:3]):
                    for skill_id in Q_mat_dict[item_id]:
                        counts = np.log(1 + np.array(counters[user_id, skill_id].get_counters(timestamp)))
                        attempts[i, skill_id * NUM_TIME_WINDOWS:(skill_id + 1) * NUM_TIME_WINDOWS] = counts
                        counters[user_id, skill_id].push(timestamp)
            elif tw == "tw_items":
                attempts = np.zeros((num_items_user, NUM_TIME_WINDOWS))
                for i, (item_id, timestamp) in enumerate(df_user[:, 1:3]):
                    attempts[i] = np.log(1 + np.array(counters[user_id, item_id].get_counters(timestamp)))
                    counters[user_id, item_id].push(timestamp)
            else:
                attempts = np.multiply(np.cumsum(np.vstack((np.zeros(num_skills), skills)), 0)[:-1], skills)
            features['attempts'] = sparse.vstack([features['attempts'], sparse.csr_matrix(attempts)])

        if "wins" in active_features:
            skills = Q_mat[df_user[:, 1].astype(int)].copy()
            if tw == "tw_kc":
                wins = np.zeros((num_items_user, NUM_TIME_WINDOWS * num_skills))
                for i, (item_id, timestamp, correct) in enumerate(df_user[:, 1:4]):
                    for skill_id in Q_mat_dict[item_id]:
                        counts = np.log(1 + np.array(counters[user_id, skill_id, "correct"].get_counters(timestamp)))
                        wins[i, skill_id * NUM_TIME_WINDOWS:(skill_id + 1) * NUM_TIME_WINDOWS] = counts
                        if correct:
                            counters[user_id, skill_id, "correct"].push(timestamp)
            elif tw == "tw_items":
                wins = np.zeros((num_items_user, NUM_TIME_WINDOWS))
                for i, (item_id, timestamp, correct) in enumerate(df_user[:, 1:4]):
                    wins[i] = np.log(1 + np.array(counters[user_id, item_id, "correct"].get_counters(timestamp)))
                    if correct:
                        counters[user_id, item_id, "correct"].push(timestamp)
            else:
                wins = np.multiply(np.cumsum(np.multiply(np.vstack((np.zeros(num_skills), skills)),
                    np.hstack((np.array([0]), df_user[:, 3])).reshape(-1, 1)), 0)[:-1], skills)
            features['wins'] = sparse.vstack([features['wins'], sparse.csr_matrix(wins)])

        if "fails" in active_features:
            skills = Q_mat[df_user[:, 1].astype(int)].copy()
            fails = np.multiply(np.cumsum(np.multiply(np.vstack((np.zeros(num_skills), skills)),
                np.hstack((np.array([0]), 1 - df_user[:, 3])).reshape(-1, 1)), 0)[:-1], skills)
            features["fails"] = sparse.vstack([features["fails"], sparse.csr_matrix(fails)])

    # User and item one hot encodings
    if 'users' in active_features:
        features['users'] = onehot.fit_transform(features["df"][:, 0].reshape(-1, 1))
    if 'items' in active_features:
        features['items'] = onehot.fit_transform(features["df"][:, 1].reshape(-1, 1))

    X = sparse.hstack([sparse.csr_matrix(features['df']),
                       sparse.hstack([features[x] for x in active_features])]).tocsr()
    return X


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode datasets.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--users', action='store_true')
    parser.add_argument('--items', action='store_true')
    parser.add_argument('--skills', action='store_true')
    parser.add_argument('--wins', action='store_true')
    parser.add_argument('--fails', action='store_true')
    parser.add_argument('--attempts', action='store_true')
    parser.add_argument('--tw_kc', action='store_true')
    parser.add_argument('--tw_items', action='store_true')
    args = parser.parse_args()

    data_path = os.path.join('data', args.dataset)

    all_features = ['users', 'items', 'skills', 'wins', 'fails', 'attempts']
    active_features = [features for features in all_features if vars(args)[features]]
    features_suffix = ''.join([features[0] for features in active_features])

    if vars(args)["tw_kc"]:
        features_suffix += 't1'
        tw = "tw_kc"
    elif vars(args)["tw_items"]:
        features_suffix += 't2'
        tw = "tw_items"
    else:
        tw = None

    df = pd.read_csv(os.path.join(data_path, 'preprocessed_data.csv'), sep="\t")
    qmat = sparse.load_npz(os.path.join(data_path, 'q_mat.npz')).toarray()
    features = df_to_sparse(df, qmat, active_features, tw=tw)
    sparse.save_npz(os.path.join(data_path, f"X-{features_suffix}.npz"), features)