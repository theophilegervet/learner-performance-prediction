import pandas as pd
import os



if __name__ == '__main__':

    summary_df_dict = {}
    dist_df_dict = {}

    for dataset_name in os.listdir('./data/'):
        print(dataset_name)
        if dataset_name in {'ednet', 'ednet_medium', 'assistments12'}:
            continue
        dataset_train = pd.read_csv('./data/{}/'.format(dataset_name) + 'preprocessed_data_train.csv', sep="\t")
        dataset_test = pd.read_csv('./data/{}/'.format(dataset_name) + 'preprocessed_data_test.csv', sep="\t")
        dataset_total = pd.concat([dataset_train, dataset_test], axis=0)
        dataset = dataset_total
        summary_dict = {}
        dist_dict = {}

        # Group Analysis
        for groupby in ['user_id', 'item_id', 'skill_id', ['user_id', 'item_id'], \
            ['user_id', 'skill_id'], ['skill_id', 'item_id']]:

            group_key = groupby.replace('_id', '') if isinstance(groupby, str) \
                else '({},{})'.format(groupby[0].replace('_id', ''), groupby[1].replace('_id', ''))
            per_key_distribution = dataset.groupby(by=groupby).apply(len)
            org_distribution = per_key_distribution.copy()
            dist_dict[group_key] = per_key_distribution
            if isinstance(groupby, list):
                per_key_distribution = per_key_distribution.unstack(0).fillna(0).stack(0)
            dist_stat = per_key_distribution.describe()
            dist_stat.loc['density'] = len(per_key_distribution.loc[per_key_distribution > 0])/len(per_key_distribution)
            dist_stat.loc['5%'] = per_key_distribution.quantile(0.05)
            dist_stat.loc['95%'] = per_key_distribution.quantile(0.95)
            summary_dict['#{}_stat'.format(group_key)] = dist_stat
            if dist_stat['density'] < 1:
                org_dist_stat = org_distribution.describe()
                org_dist_stat.loc['5%'] = org_distribution.quantile(0.05)
                org_dist_stat.loc['95%'] = org_distribution.quantile(0.95)
                summary_dict['#{}_nonzero_stat'.format(group_key)] = org_dist_stat

        stat_keys = [key for key in summary_dict.keys() if key.endswith('_stat')]
        rename_keys = [key.replace('#', '').replace('_stat', ' data') for key in stat_keys]
        summary_table = pd.concat([summary_dict[key] for key in stat_keys], axis=1, \
            keys=rename_keys).T.round(3)
        
        summary_table.columns = ['unique count', 'avg', 'std', '0%(min)', '25%', \
            '50%(med)', '75%', '100%(max)', 'matrix density', '5%', '95%']
        summary_table = summary_table[['unique count', 'avg', '0%(min)', '5%', '25%', \
            '50%(med)', '75%', '95%', '100%(max)', 'matrix density']]
        print(summary_table)

        summary_df_dict[dataset_name] = summary_table
        dist_df_dict[dataset_name] = dist_dict
    
    all_df = pd.concat([y for _, y in summary_df_dict.items()], axis=0, keys=summary_df_dict.keys())
    all_df.index.names = ['dataset_name', 'field']
    all_df.sort_index(axis=0, level='field', inplace=True, ascending=False )
    all_df.to_csv('./dataset summary.csv')