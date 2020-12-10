import pandas as pd
import numpy as np
import pickle


if __name__ == '__main__':
    with open('/root/lpp/data/ednet/tcr_user_split.pkl', 'rb') as file:
        ednet_split = pickle.load(file)

    ednet_data_path = '/root/lpp/data/ednet/prep_tcr.csv'
    ednet_data = pd.read_csv(ednet_data_path)
    row = 0
    row_ind = 0
    row_per_file = 500000
    while row < ednet_data.shape[0]:
        ednet_data_partition = ednet_data.loc[row:min(row + row_per_file, ednet_data.shape[0])]
        ednet_data_partition['correct'] = (ednet_data_partition['user_answer'] == ednet_data_partition['correct_answer'])
        ednet_timestamp = ednet_data_partition['start_time'].apply(pd.to_datetime)
        ednet_timestamp = (ednet_timestamp - ednet_timestamp.min())
        ednet_timestamp = ednet_timestamp.apply(lambda x: x.total_seconds()).astype(np.int64)
        ednet_data_partition['timestamp'] = ednet_timestamp
        ednet_slim = ednet_data_partition[['student_id', 'timestamp', 'content_id', 'correct']]
        ednet_slim.columns = ['user_id', 'timestamp', 'item_id', 'correct']
        ednet_slim.to_csv('./data/ednet/preprocessed_data_split_{}.csv'.format(row_ind))

        print(row_ind)
        row_ind += 1
        row += row_per_file

    del ednet_data

    data_list = []
    for row_ind_ in range(row_ind):
        partition_data = pd.read_csv('./data/ednet/preprocessed_data_split{}.csv'.format(row_ind_))
        data_list.append(partition_data)
    
    pd.concat(data_list, axis=1).to_csv('./data/ednet/preprocessed_data.csv')