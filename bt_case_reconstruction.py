import pandas as pd
import numpy as np
import random


def test_seq_reconstruction(
    data_df, 
    item_or_skill='item',
    min_sample_num=3, 
    min_thres=1,
    max_delay=np.inf,  #TODO
    ):
    user_key_sample_len = {}
    data_df['testpoint'] = np.nan
    keycol = '{}_id'.format(item_or_skill)
    for (user_id, key_id), user_key_df in data_df.groupby(['user_id', keycol]):
        if len(user_key_df) < min_sample_num:
            continue
        user_key_sample_len[(user_id, key_id)] = len(user_key_df)
        expand_win_avg = user_key_df['correct'].expanding(min_periods=min_sample_num).mean()
        test_points = expand_win_avg.loc[expand_win_avg.subtract(0.5).abs() >= abs(min_thres - 0.5)]
        if len (test_points):
            data_df.loc[test_points.index, 'testpoint'] = test_points.round()
   
    test_df_list = []
    new_user_id = data_df['user_id'].max() + 1
    for test_row in data_df.loc[data_df['testpoint'].notnull()].index:
        test_id = data_df[keycol][test_row]
        user_id = data_df['user_id'][test_row]
        user_df = data_df.loc[data_df['user_id'] == user_id]
        pre_df = user_df.loc[user_df.index <= test_row]
        post_df = user_df.loc[user_df.index > test_row]
        if test_id in post_df[keycol].unique():
            post_df = post_df.loc[:(post_df[keycol] == test_id).idxmax()].iloc[:-1]
        test_interaction = data_df.loc[[test_row]].copy()
        test_interaction['correct'] = test_interaction['testpoint']
        test_interaction['timestamp'] = np.nan
        # insert virtual test interaction into post_df
        post_df.reset_index(drop=True, inplace=True)
        post_df.index = post_df.index + 1
        insert_index = random.sample(range(post_df.shape[0] + 1), 1)[0]
        new_post_df = pd.concat(
            [post_df.loc[:insert_index],
            test_interaction], axis=0)
        new_df = pd.concat([
            pre_df.reset_index(drop=True),
            new_post_df.reset_index(drop=True)
            ], axis=0
        ).reset_index(drop=True)
        new_df['user_id'] = new_user_id
        new_df['timestamp'] = new_df['timestamp'].ffill()
        new_user_id += 1
        test_df_list.append(new_df)
    
    new_data = pd.concat(test_df_list, axis=0).reset_index(drop=True)
    data_meta = {
        'num_sample': new_data['user_id'].unique().shape[0],
        'num_interaction': new_data.shape[0],
    }
    return new_data, data_meta
