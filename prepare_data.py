import numpy as np
import pandas as pd
from scipy import sparse
import argparse
import os


def prepare_assistments(data_name, min_interactions_per_user, remove_nan_skills):
    """Preprocess ASSISTments dataset.
    
    Arguments:
        data_name: "assistments09", "assistments12", "assistments15" or "assistments17"
        min_interactions_per_user (int): minimum number of interactions per student
        remove_nan_skills (bool): if True, remove interactions with no skill tag

    Outputs:
        df (pandas DataFrame): preprocessed ASSISTments dataset with user_id, item_id,
            timestamp and correct features
        Q_mat (item-skill relationships sparse array): corresponding q-matrix
    """
    data_path = os.path.join("data", data_name)
    df = pd.read_csv(os.path.join(data_path, "data.csv"), encoding="ISO-8859-1")

    # Only 2012 and 2017 versions have timestamps
    if data_name == "assistments09":
        df = df.rename(columns={"problem_id": "item_id"})
        df["timestamp"] = np.zeros(len(df), dtype=np.int64)
    elif data_name == "assistments12":
        df = df.rename(columns={"problem_id": "item_id"})
        df["timestamp"] = pd.to_datetime(df["start_time"])
        df["timestamp"] = df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
        df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    elif data_name == "assistments15":
        df = df.rename(columns={"sequence_id": "item_id"})
        df["skill_id"] = df["item_id"]
        df["timestamp"] = np.zeros(len(df), dtype=np.int64)
    elif data_name == "assistments17":
        df = df.rename(columns={"startTime": "timestamp",
                                "studentId": "user_id",
                                "problemId": "item_id",
                                "skill": "skill_id"})
        df["timestamp"] = df["timestamp"] - df["timestamp"].min()

    # Sort data temporally
    if data_name in ["assistments12", "assistments17"]:
        df.sort_values(by="timestamp", inplace=True)
    elif data_name == "assistments09":
        df.sort_values(by="order_id", inplace=True)
    elif data_name == "assistments15":
        df.sort_values(by="log_id", inplace=True)

    # Filter too short sequences
    df = df.groupby("user_id").filter(lambda x: len(x) >= min_interactions_per_user)

    # Remove continuous outcomes
    df = df[df["correct"].isin([0, 1])]

    # Filter nan skills
    if remove_nan_skills:
        df = df[~df["skill_id"].isnull()]
    else:
        df.ix[df["skill_id"].isnull(), "skill_id"] = -1

    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]
    df["skill_id"] = np.unique(df["skill_id"], return_inverse=True)[1]

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(df["skill_id"].unique())))
    for item_id, skill_id in df[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1

    df = df[['user_id', 'item_id', 'timestamp', 'correct']]
    df["correct"] = df["correct"].astype(np.int32)
    df.reset_index(inplace=True, drop=True)

    # Save data
    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), sparse.csr_matrix(Q_mat))
    df.to_csv(os.path.join(data_path, "preprocessed_data.csv"), sep="\t", index=False)


def prepare_kddcup10(data_name, min_interactions_per_user, kc_col_name, remove_nan_skills):
    """Preprocess KDD Cup 2010 dataset.

    Arguments:
        data_name (str): "bridge_algebra06" or "algebra05"
        min_interactions_per_user (int): minimum number of interactions per student
        kc_col_name (str): Skills id column
        remove_nan_skills (bool): if True, remove interactions with no skill tag

    Outputs:
        df (pandas DataFrame): preprocessed KDD Cup 2010 dataset with user_id, item_id,
            timestamp and correct features
        Q_mat (item-skill relationships sparse array): corresponding q-matrix
    """
    data_path = os.path.join("data", data_name)
    df = pd.read_csv(os.path.join(data_path, "data.txt"), delimiter='\t')
    df = df.rename(columns={'Anon Student Id': 'user_id',
                            'Correct First Attempt': 'correct'})

    # Create item from problem and step
    df["item_id"] = df["Problem Name"] + ":" + df["Step Name"]

    # Add timestamp
    df["timestamp"] = pd.to_datetime(df["First Transaction Time"])
    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    df["timestamp"] = df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
    df.sort_values(by="timestamp", inplace=True)

    # Filter too short sequences
    df = df.groupby("user_id").filter(lambda x: len(x) >= min_interactions_per_user)

    # Remove continuous outcomes
    df = df[df["correct"].isin([0, 1])]

    # Filter nan skills
    if remove_nan_skills:
        df = df[~df[kc_col_name].isnull()]
    else:
        df.ix[df[kc_col_name].isnull(), kc_col_name] = 'NaN'

    # Drop duplicates
    df.drop_duplicates(subset=["user_id", "item_id", "timestamp"], inplace=True)

    # Extract KCs
    kc_list = []
    for kc_str in df[kc_col_name].unique():
        for kc in kc_str.split('~~'):
            kc_list.append(kc)
    kc_set = set(kc_list)
    kc2idx = {kc: i for i, kc in enumerate(kc_set)}

    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(kc_set)))
    for item_id, kc_str in df[["item_id", kc_col_name]].values:
        for kc in kc_str.split('~~'):
            Q_mat[item_id, kc2idx[kc]] = 1

    df = df[['user_id', 'item_id', 'timestamp', 'correct']]
    df['correct'] = df['correct'].astype(np.int32)
    df.reset_index(inplace=True, drop=True)

    # Save data
    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), sparse.csr_matrix(Q_mat))
    df.to_csv(os.path.join(data_path, "preprocessed_data.csv"), sep="\t", index=False)


def prepare_squirrel_ai(min_interactions_per_user):
    """Preprocess Squirrel AI dataset.

    Arguments:
        min_interactions_per_user (int): minimum number of interactions per student

    Outputs:
        df (pandas DataFrame): preprocessed Squirrel AI dataset with user_id, item_id,
            timestamp and correct features
        Q_mat (item-skill relationships sparse array): corresponding q-matrix
    """
    data_path = "data/squirrel-ai"

    train_df = pd.read_csv(os.path.join(data_path, "studentDataFIT.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "studentDataTEST.csv"))
    train_df, test_df = [df.rename(columns={"student_index": "user_id",
                                            "question_index": "item_id",
                                            "KP_index": "skill_id",
                                            "is_correct": "correct"})
                         for df in (train_df, test_df)]


    for i, df in enumerate([train_df, test_df]):
        # Timestamp in seconds
        df["timestamp"] = df["decimalTimeAnswered"] * 3600 * 24
        df["timestamp"] = (df["timestamp"] - df["timestamp"].min()).astype(np.int64)

        # Filter too short sequences
        df = df.groupby("user_id").filter(lambda x: len(x) >= min_interactions_per_user)

        if i == 0:
            df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
        else:
            df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1] + train_df["user_id"].nunique()

        # Build Q-matrix
        num_items = max(train_df["item_id"].max(), test_df["item_id"].max()) + 1
        num_skills = max(train_df["skill_id"].max(), test_df["skill_id"].max()) + 1
        Q_mat = np.zeros((num_items, num_skills))
        for item_id, skill_id in df[["item_id", "skill_id"]].values:
            Q_mat[item_id, skill_id] = 1

        df = df[['user_id', 'item_id', 'timestamp', 'correct']]
        df["correct"] = df["correct"].astype(np.int32)
        df.reset_index(inplace=True, drop=True)

        # Save data
        suffix = "train" if i == 0 else "test"
        sparse.save_npz(os.path.join(data_path, f"q_mat_{suffix}.npz"), sparse.csr_matrix(Q_mat))
        df.to_csv(os.path.join(data_path, f"preprocessed_data_{suffix}.csv"), sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare datasets.')
    parser.add_argument('--dataset', type=str, default='assistments12')
    parser.add_argument('--min_interactions', type=int, default=10)
    parser.add_argument('--remove_nan_skills', action='store_true')
    args = parser.parse_args()

    if args.dataset in ["assistments09", "assistments12", "assistments15", "assistments17"]:
        prepare_assistments(
            data_name=args.dataset,
            min_interactions_per_user=args.min_interactions,
            remove_nan_skills=args.remove_nan_skills)
    elif args.dataset == "bridge_algebra06":
        prepare_kddcup10(
            data_name="bridge_algebra06",
            min_interactions_per_user=args.min_interactions,
            kc_col_name="KC(SubSkills)",
            remove_nan_skills=args.remove_nan_skills)
    elif args.dataset == "algebra05":
        prepare_kddcup10(
            data_name="algebra05",
            min_interactions_per_user=args.min_interactions,
            kc_col_name="KC(Default)",
            remove_nan_skills=args.remove_nan_skills)
    elif args.dataset == "squirrel-ai":
        prepare_squirrel_ai(
            min_interactions_per_user=args.min_interactions)