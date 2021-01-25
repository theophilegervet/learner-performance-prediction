import pandas as pd
import random

def gen_repeated_feed(
    data_df, 
    item_or_skill='item',
    repeat_val_list=[1, 0],
    repeat_length=10
    ):
    if item_or_skill == 'skill':
        raise NotImplementedError
    item2skill = data_df.groupby('item_id').first()['skill_id']
    df_list = []
    sorted_timestamps = data_df['timestamp'].sort_values()
    for item_id in data_df[f'{item_or_skill}_id'].unique():
        for repeat_val in repeat_val_list:
            content_val_row = pd.Series({
                'user_id': item_id + repeat_val * data_df['user_id'].max(), 
                'item_id': item_id,
                'skill_id': item2skill[item_id],
                'correct': repeat_val,
            })
            content_val_df = pd.concat([content_val_row.to_frame().T \
                for _ in range(repeat_length)], axis=0).reset_index(drop=True)
            content_val_df['timestamp'] = sorted_timestamps.iloc[
                random.sample(list(range(len(sorted_timestamps))), repeat_length)].values
            df_list.append(content_val_df)
    total_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    total_df['testpoint'] = total_df['correct']
    return total_df, {}
