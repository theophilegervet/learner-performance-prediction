import argparse
import pandas as pd
import numpy as np
import random
import torch

from models.model_dkt2 import DKT2
from models.model_sakt2 import SAKT

from train_dkt2 import get_data, prepare_batches, eval_batches
from train_saint import SAINT, DataModule, predict_saint

from testcase_template import (
    df_perturbation,
    perturb_review_step,
    perturb_add_last,
    perturb_add_last_random
)
from utils import *
import pytorch_lightning as pl


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


def test_repeated_feed(
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioral Testing")
    parser.add_argument("--dataset", type=str, default="ednet_small")
    parser.add_argument("--model", type=str, \
        choices=["lr", "dkt", "sakt", "saint"], default="saint")
    parser.add_argument("--test_type", type=str, default="reconstruction")
    parser.add_argument("--load_dir", type=str, default="./save/")
    parser.add_argument("--filename", type=str, default="ednet_small")
    parser.add_argument("--gpu", type=str, default="0,1")
    parser.add_argument("--diff_threshold", type=float, default=0.05)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # LOAD MODEL - SAINT OR {DKT, BESTLR, SAKT}
    if args.model == 'saint':
        import pickle
        checkpoint_path = f'./save/{args.model}/' + args.filename + '.ckpt'
        with open(checkpoint_path.replace('.ckpt', '_config.pkl'), 'rb') as file:
            saint_config = argparse.Namespace(**pickle.load(file))
        model = SAINT.load_from_checkpoint(checkpoint_path, config=saint_config\
            ).to(torch.device("cuda"))
        model.eval()
    else:
        saver = Saver(args.load_dir + f'/{args.model}/', args.filename)
        model = saver.load().to(torch.device("cuda"))
        model.eval()

    test_df = pd.read_csv(
        os.path.join("data", args.dataset, "preprocessed_data_test.csv"), sep="\t"
    )

    # Generate new test data based on test type.
    last_one_only = False
    if args.test_type == 'reconstruction':
        bt_test_df, new_test_meta = test_seq_reconstruction(test_df, item_or_skill='item')
        last_one_only = True
    elif args.test_type == 'repetition':
        bt_test_df, new_test_meta = test_repeated_feed(test_df, item_or_skill='item')
    elif args.test_type == 'add_last':
        bt_test_df, new_test_meta = df_perturbation(test_df, perturb_add_last_random)
    elif args.test_type == 'deletion':
        pass
    elif args.test_type == 'replacement':
        pass
    else:
        raise NotImplementedError("Not implemented test_type")

    # Generate model output.
    bt_test_path = os.path.join("data", args.dataset, "bt_{}.csv".format(args.test_type))
    bt_test_df.to_csv(bt_test_path)
    if args.model == 'saint':
        datamodule = DataModule(saint_config, overwrite_test_df=bt_test_df, last_one_only=last_one_only)
        trainer = pl.Trainer(auto_select_gpus=True, callbacks=[], max_steps=0)
        bt_test_preds = predict_saint(saint_model=model, dataloader=datamodule.test_dataloader())
        if last_one_only:
            sub_df = bt_test_df.groupby('user_id').last()
        else:
            sub_df = bt_test_df
        sub_df['model_pred'] = bt_test_preds.cpu()
    else:
        bt_test_data, _ = get_data(bt_test_df, train_split=1.0, randomize=False)
        bt_test_batch = prepare_batches(bt_test_data, 10, False)
        bt_test_preds = eval_batches(model, bt_test_batch, 'cuda')
        bt_test_df['model_pred'] = bt_test_preds
        if last_one_only:
            sub_df = bt_test_df.groupby('user_id').last()
        else:
            sub_df = bt_test_df

    # Check test constraints.
    if args.test_type == 'reconstruction':
        sub_df['testpass'] = (sub_df['testpoint'] == sub_df['model_pred'].round())
        sub_df.to_csv('./bt_result_{}.csv'.format(args.test_type))
        print(sub_df['testpass'].describe())
        print(sub_df.loc[sub_df['testpoint'] == 0, 'testpass'].describe())
        print(sub_df.loc[sub_df['testpoint'] == 1, 'testpass'].describe())

    elif args.test_type == 'repetition':
        sub_df['testpass'] = (sub_df['testpoint'] == sub_df['model_pred'].round())
        sub_df.to_csv('./bt_result_{}.csv'.format(args.test_type))
        print(sub_df['testpass'].describe())
        print(sub_df.loc[sub_df['testpoint'] == 0, 'testpass'].describe())
        print(sub_df.loc[sub_df['testpoint'] == 1, 'testpass'].describe())

    elif args.test_type == 'add_last':
        user_group_df = sub_df.groupby('orig_user_id')
        user_group_df['testpass'] = False
        for name, group in user_group_df:
            orig_prob = group.loc[group['is_perturbed'] == 0]['model_pred'].item()
            corr_prob = group.loc[group['is_perturbed'] == 1]['model_pred'].item()
            incorr_prob = group.loc[group['is_perturbed'] == -1]['model_pred'].item()
            if corr_prob >= orig_prob - args.diff_threshold:
                user_group_df.loc[
                    (user_group_df['orig_user_id'] == name) & (user_group_df['is_perturbed'] == 1),
                    'testpass'] = True
            if incorr_prob <= orig_prob + args.diff_threshold:
                user_group_df.loc[
                    (user_group_df['orig_user_id'] == name) & (user_group_df['is_perturbed'] == 1),
                    'testpass'] = True
        print(user_group_df.loc[user_group_df['is_perturbed'] != 0, 'testpass'].describe())
        print(user_group_df.loc[user_group_df['is_perturbed'] == 1, 'testpass'].describe())
        print(user_group_df.loc[user_group_df['is_perturbed'] == -1, 'testpass'].describe())
    
    elif args.test_type == 'deletion':
        pass
    elif args.test_type == 'replacement':
        pass

