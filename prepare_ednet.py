import os
import pandas as pd
import numpy as np
import pickle


def limit_max_seq(data_df, max_seq_length=100, stride=50):
    """
    Only for training dataset.
    """
    data_df["user_id"]

    pass


if __name__ == "__main__":
    with open("/root/lpp/data/ednet/tcr_user_split.pkl", "rb") as file:
        ednet_split = pickle.load(file)

    if 0:
        ednet_data_path = "/root/lpp/data/ednet/prep_tcr.csv"
        ednet_data = pd.read_csv(ednet_data_path)
        row = 0
        row_ind = 0
        row_per_file = 500000
        while row < ednet_data.shape[0]:
            split_file_path = "./data/ednet/preprocessed_data_split_{}.csv".format(
                row_ind
            )
            if not os.path.exists(split_file_path):
                ednet_data_partition = ednet_data.loc[
                    row : min(row + row_per_file, ednet_data.shape[0])
                ]
                ednet_data_partition["correct"] = (
                    ednet_data_partition["user_answer"]
                    == ednet_data_partition["correct_answer"]
                )
                ednet_timestamp = ednet_data_partition["start_time"].apply(
                    pd.to_datetime
                )
                ednet_timestamp = ednet_timestamp - ednet_timestamp.min()
                ednet_timestamp = ednet_timestamp.apply(
                    lambda x: x.total_seconds()
                ).astype(np.int64)
                ednet_data_partition["timestamp"] = ednet_timestamp
                ednet_slim = ednet_data_partition[
                    ["student_id", "timestamp", "content_id", "correct"]
                ]
                ednet_slim.columns = ["user_id", "timestamp", "item_id", "correct"]
                ednet_slim.to_csv(split_file_path)
            else:
                print("skipping ", row_ind)
                pass

            print(row_ind)
            row_ind += 1
            row += row_per_file

        del ednet_data

        data_list = []
        for row_ind_ in range(194):
            partition_data = pd.read_csv(
                "./data/ednet/preprocessed_data_split_{}.csv".format(row_ind_)
            )
            data_list.append(partition_data)

        pd.concat(data_list, axis=0).to_csv("./data/ednet/preprocessed_data.csv")

    data_path = "/root/lpp/data/ednet/"
    ednet_data = pd.read_csv(data_path + "preprocessed_data.csv")[
        ["user_id", "item_id", "timestamp", "correct"]
    ].set_index(["user_id", "timestamp"])
    ind2type = {0: "train", 1: "valid", 2: "test"}
    for ind, type in ind2type.items():
        ednet_data.loc[ednet_split[ind]].reset_index().to_csv(
            data_path + "preprocessed_data_{}.csv".format(type)
        )
