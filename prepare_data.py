import numpy as np
import pandas as pd
from scipy import sparse
import argparse
import os


def prepare_assistments(data_name, min_interactions_per_user, remove_nan_skills, train_split=0.8):
    """Preprocess ASSISTments dataset.
    
    Arguments:
        data_name: "assistments09", "assistments12", "assistments15" or "assistments17"
        min_interactions_per_user (int): minimum number of interactions per student
        remove_nan_skills (bool): if True, remove interactions with no skill tag
        train_split (float): proportion of data to use for training

    Outputs:
        df (pandas DataFrame): preprocessed ASSISTments dataset with user_id, item_id,
            timestamp, correct and unique skill features
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
        df["timestamp"] = df["timestamp"] - df["timestamp"].min()
        df["timestamp"] = df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
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

    # Remove continuous outcomes
    df = df[df["correct"].isin([0, 1])]
    df["correct"] = df["correct"].astype(np.int32)

    # Filter nan skills
    if remove_nan_skills:
        df = df[~df["skill_id"].isnull()]
    else:
        df.ix[df["skill_id"].isnull(), "skill_id"] = -1

    # Filter too short sequences
    df = df.groupby("user_id").filter(lambda x: len(x) >= min_interactions_per_user)

    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]
    df["skill_id"] = np.unique(df["skill_id"], return_inverse=True)[1]

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(df["skill_id"].unique())))
    for item_id, skill_id in df[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1

    # Remove row duplicates due to multiple skills for one item
    if data_name == "assistments09":
        df = df.drop_duplicates("order_id")
    elif data_name == "assistments17":
        df = df.drop_duplicates(["user_id", "timestamp"])

    # Get unique skill id from combination of all skill ids
    unique_skill_ids = np.unique(Q_mat, axis=0, return_inverse=True)[1]
    df["skill_id"] = unique_skill_ids[df["item_id"]]

    # Sort data temporally
    if data_name in ["assistments12", "assistments17"]:
        df.sort_values(by="timestamp", inplace=True)
    elif data_name == "assistments09":
        df.sort_values(by="order_id", inplace=True)
    elif data_name == "assistments15":
        df.sort_values(by="log_id", inplace=True)

    # Sort data by users, preserving temporal order for each user
    df = pd.concat([u_df for _, u_df in df.groupby("user_id")])

    df = df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]
    df.reset_index(inplace=True, drop=True)

    # Text files for BKT implementation (https://github.com/robert-lindsey/WCRP/)
    bkt_dataset = df[["user_id", "item_id", "correct"]]
    bkt_skills = unique_skill_ids
    bkt_split = np.random.randint(low=0, high=5, size=df["user_id"].nunique()).reshape(1, -1)

    # Train-test split
    users = df["user_id"].unique()
    np.random.shuffle(users)
    split = int(train_split * len(users))
    train_df = df[df["user_id"].isin(users[:split])]
    test_df = df[df["user_id"].isin(users[split:])]

    # Save data
    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), sparse.csr_matrix(Q_mat))
    train_df.to_csv(os.path.join(data_path, "preprocessed_data_train.csv"), sep="\t", index=False)
    test_df.to_csv(os.path.join(data_path, "preprocessed_data_test.csv"), sep="\t", index=False)
    df.to_csv(os.path.join(data_path, "preprocessed_data.csv"), sep="\t", index=False)
    np.savetxt(os.path.join(data_path, "bkt_dataset.txt"), bkt_dataset, fmt='%i')
    np.savetxt(os.path.join(data_path, "bkt_expert_labels.txt"), bkt_skills, fmt='%i')
    np.savetxt(os.path.join(data_path, "bkt_splits.txt"), bkt_split, fmt='%i')


def prepare_kddcup10(data_name, min_interactions_per_user, kc_col_name, remove_nan_skills, train_split=0.8):
    """Preprocess KDD Cup 2010 dataset.

    Arguments:
        data_name (str): "bridge_algebra06" or "algebra05"
        min_interactions_per_user (int): minimum number of interactions per student
        kc_col_name (str): Skills id column
        remove_nan_skills (bool): if True, remove interactions with no skill tag
        train_split (float): proportion of data to use for training

    Outputs:
        df (pandas DataFrame): preprocessed KDD Cup 2010 dataset with user_id, item_id,
            timestamp, correct and unique skill features
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

    # Remove continuous outcomes
    df = df[df["correct"].isin([0, 1])]
    df['correct'] = df['correct'].astype(np.int32)

    # Filter nan skills
    if remove_nan_skills:
        df = df[~df[kc_col_name].isnull()]
    else:
        df.ix[df[kc_col_name].isnull(), kc_col_name] = 'NaN'

    # Drop duplicates
    df.drop_duplicates(subset=["user_id", "item_id", "timestamp"], inplace=True)

    # Filter too short sequences
    df = df.groupby("user_id").filter(lambda x: len(x) >= min_interactions_per_user)

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

    # Get unique skill id from combination of all skill ids
    unique_skill_ids = np.unique(Q_mat, axis=0, return_inverse=True)[1]
    df["skill_id"] = unique_skill_ids[df["item_id"]]

    # Sort data temporally
    df.sort_values(by="timestamp", inplace=True)

    # Sort data by users, preserving temporal order for each user
    df = pd.concat([u_df for _, u_df in df.groupby("user_id")])

    df = df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]
    df.reset_index(inplace=True, drop=True)

    # Text files for BKT implementation (https://github.com/robert-lindsey/WCRP/)
    bkt_dataset = df[["user_id", "item_id", "correct"]]
    bkt_skills = unique_skill_ids
    bkt_split = np.random.randint(low=0, high=5, size=df["user_id"].nunique()).reshape(1, -1)

    # Train-test split
    users = df["user_id"].unique()
    np.random.shuffle(users)
    split = int(train_split * len(users))
    train_df = df[df["user_id"].isin(users[:split])]
    test_df = df[df["user_id"].isin(users[split:])]

    # Save data
    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), sparse.csr_matrix(Q_mat))
    train_df.to_csv(os.path.join(data_path, "preprocessed_data_train.csv"), sep="\t", index=False)
    test_df.to_csv(os.path.join(data_path, "preprocessed_data_test.csv"), sep="\t", index=False)
    df.to_csv(os.path.join(data_path, "preprocessed_data.csv"), sep="\t", index=False)
    np.savetxt(os.path.join(data_path, "bkt_dataset.txt"), bkt_dataset, fmt='%i')
    np.savetxt(os.path.join(data_path, "bkt_expert_labels.txt"), bkt_skills, fmt='%i')
    np.savetxt(os.path.join(data_path, "bkt_splits.txt"), bkt_split, fmt='%i')


def prepare_squirrel_ai(min_interactions_per_user):
    """Preprocess Squirrel AI dataset.

    Arguments:
        min_interactions_per_user (int): minimum number of interactions per student

    Outputs:
        df (pandas DataFrame): preprocessed Squirrel AI dataset with user_id, item_id,
            timestamp, correct and unique skill features
        Q_mat (item-skill relationships sparse array): corresponding q-matrix
    """
    data_path = "data/squirrel_ai"

    train_df = pd.read_csv(os.path.join(data_path, "studentDataFIT.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "studentDataTEST.csv"))

    train_df, test_df = [df.rename(columns={"student_index": "user_id",
                                            "question_index": "item_id",
                                            "KP_index": "skill_id",
                                            "is_correct": "correct"})
                         for df in (train_df, test_df)]

    # Timestamp in seconds
    train_df["timestamp"] = train_df["decimalTimeAnswered"] * 3600 * 24
    train_df["timestamp"] = (train_df["timestamp"] - train_df["timestamp"].min()).astype(np.int64)
    test_df["timestamp"] = test_df["decimalTimeAnswered"] * 3600 * 24
    test_df["timestamp"] = (test_df["timestamp"] - test_df["timestamp"].min()).astype(np.int64)

    # Filter too short sequences
    train_df = train_df.groupby("user_id").filter(lambda x: len(x) >= min_interactions_per_user)
    test_df = test_df.groupby("user_id").filter(lambda x: len(x) >= min_interactions_per_user)

    train_df["user_id"] = np.unique(train_df["user_id"], return_inverse=True)[1]
    test_df["user_id"] = np.unique(test_df["user_id"], return_inverse=True)[1] + train_df["user_id"].nunique()

    # Build Q-matrix
    num_items = max(train_df["item_id"].max(), test_df["item_id"].max()) + 1
    num_skills = max(train_df["skill_id"].max(), test_df["skill_id"].max()) + 1
    Q_mat = np.zeros((num_items, num_skills))
    for df in (train_df, test_df):
        for item_id, skill_id in df[["item_id", "skill_id"]].values:
            Q_mat[item_id, skill_id] = 1

    # Get unique skill id from combination of all skill ids
    unique_skill_ids = np.unique(Q_mat, axis=0, return_inverse=True)[1]
    train_df["skill_id"] = unique_skill_ids[train_df["item_id"]]
    test_df["skill_id"] = unique_skill_ids[test_df["item_id"]]

    # Data is already sorted by users and temporally for each user
    train_df = train_df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]
    test_df = test_df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]
    df = pd.concat([train_df, test_df])
    train_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
    df.reset_index(inplace=True, drop=True)

    # Text files for BKT implementation (https://github.com/robert-lindsey/WCRP/)
    bkt_dataset = df[["user_id", "item_id", "correct"]]
    bkt_skills = unique_skill_ids
    bkt_split = np.random.randint(low=0, high=5, size=df["user_id"].nunique()).reshape(1, -1)

    # Save data
    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), sparse.csr_matrix(Q_mat))
    train_df.to_csv(os.path.join(data_path, f"preprocessed_data_train.csv"), sep="\t", index=False)
    test_df.to_csv(os.path.join(data_path, f"preprocessed_data_test.csv"), sep="\t", index=False)
    df.to_csv(os.path.join(data_path, f"preprocessed_data.csv"), sep="\t", index=False)
    np.savetxt(os.path.join(data_path, "bkt_dataset.txt"), bkt_dataset, fmt='%i')
    np.savetxt(os.path.join(data_path, "bkt_expert_labels.txt"), bkt_skills, fmt='%i')
    np.savetxt(os.path.join(data_path, "bkt_splits.txt"), bkt_split, fmt='%i')


def prepare_spanish(train_split=0.8):
    """Preprocess Spanish dataset.

    Arguments:
        train_split (float): proportion of data to use for training

    Outputs:
        df (pandas DataFrame): preprocessed Spanish dataset with user_id, item_id,
            timestamp, correct and unique skill features
        Q_mat (item-skill relationships sparse array): corresponding q-matrix
    """
    data_path = "data/spanish"

    data = np.loadtxt(os.path.join(data_path, "spanish_dataset.txt"), dtype=int)
    df = pd.DataFrame(data=data, columns=("user_id", "item_id", "correct"))

    skills = np.loadtxt(os.path.join(data_path, "spanish_expert_labels.txt"))
    df["skill_id"] = skills[df["item_id"]].astype(np.int64)

    df["timestamp"] = np.zeros(len(df), np.int64)

    df = df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]
    df.reset_index(inplace=True, drop=True)

    # Build Q-matrix
    Q_mat = np.zeros((df["item_id"].nunique(), df["skill_id"].nunique()))
    for item_id, skill_id in df[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1

    # Sort data by users, preserving temporal order for each user
    df = pd.concat([u_df for _, u_df in df.groupby("user_id")])

    # Train-test split
    users = df["user_id"].unique()
    np.random.shuffle(users)
    split = int(train_split * len(users))
    train_df = df[df["user_id"].isin(users[:split])]
    test_df = df[df["user_id"].isin(users[split:])]

    # Save data
    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), sparse.csr_matrix(Q_mat))
    train_df.to_csv(os.path.join(data_path, "preprocessed_data_train.csv"), sep="\t", index=False)
    test_df.to_csv(os.path.join(data_path, "preprocessed_data_test.csv"), sep="\t", index=False)
    df.to_csv(os.path.join(data_path, "preprocessed_data.csv"), sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare datasets.')
    parser.add_argument('--dataset', type=str, default='assistments09')
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
    elif args.dataset == "squirrel_ai":
        prepare_squirrel_ai(
            min_interactions_per_user=args.min_interactions)
    elif args.dataset == "spanish":
        prepare_spanish()
