import argparse
import pandas as pd
import numpy as np
import random
import torch

from models.model_dkt2 import DKT2
from models.model_sakt2 import SAKT

from train_dkt2 import get_data, prepare_batches, eval_batches
from train_saint import SAINT, DataModule, predict_saint

from testcase_template import *
from utils import *
import pytorch_lightning as pl


def wrap_input(input):
    return (torch.stack((x,)) for x in input)


def test_flip_all(model, data):
    with torch.no_grad():
        test_results = []
        for single_data in data:
            (
                item_inputs,
                skill_inputs,
                label_inputs,
                item_ids,
                skill_ids,
                labels,
            ) = single_data
            orig_input = (item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            orig_output = model(*(wrap_input(orig_input)))
            orig_output = torch.sigmoid(orig_output)
            orig_output = orig_output[0][-1]
            perturb_input, pass_range_true = generate_test_case(
                orig_input, orig_output, perturb_flip_all, (1,), pass_increase
            )
            perturb_output_true = model(*(wrap_input(perturb_input)))
            perturb_output_true = torch.sigmoid(perturb_output_true)
            perturb_output_true = perturb_output_true[0][-1]
            perturb_input, pass_range_false = generate_test_case(
                orig_input, orig_output, perturb_flip_all, (0,), pass_decrease
            )
            perturb_output_false = model(*(wrap_input(perturb_input)))
            perturb_output_false = torch.sigmoid(perturb_output_false)
            perturb_output_false = perturb_output_false[0][-1]
            test_results.append(
                [
                    orig_output.item(),
                    perturb_output_true.item(),
                    float_in_range(perturb_output_true, pass_range_true).item(),
                    perturb_output_false.item(),
                    float_in_range(perturb_output_false, pass_range_false).item(),
                ]
            )
    return test_results


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioral Testing")
    parser.add_argument("--dataset", type=str, default="ednet_small")
    parser.add_argument("--model", type=str, \
        choices=["lr", "dkt", "sakt", "saint"], default="saint")
    parser.add_argument("--test_type", nargs="+", default="reconstruction")
    parser.add_argument("--load_dir", type=str, default="./save/")
    parser.add_argument("--filename", type=str, default="ednet_small")
    parser.add_argument("--gpu", type=str, default="0,1")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # LOAD MODEL - SAINT OR {DKT, BESTLR, SAKT}
    if args.model == 'saint':
        import pickle
        checkpoint_path = './save/saint/' + args.filename + '.ckpt'
        with open(checkpoint_path.replace('.ckpt', '_config.pkl'), 'rb') as file:
            saint_config = argparse.Namespace(**pickle.load(file))
        model = SAINT.load_from_checkpoint(checkpoint_path, config=saint_config\
            ).to(torch.device("cuda"))
        model.eval()
    else:
        saver = Saver(args.load_dir + f'/{args.dataset}/', args.filename)
        model = saver.load().to(torch.device("cuda"))
        model.eval()

    # testing one sample data: change first interaction
    test_df = pd.read_csv(
        os.path.join("data", args.dataset, "preprocessed_data_test.csv"), sep="\t"
    )

    if args.test_type == 'reconstruction':
        TEST_MODE = 'item'
        bt_test_path = os.path.join("data", args.dataset, "bt_reconstruct_{}.csv".format(TEST_MODE))
        bt_test_df, new_test_meta = test_seq_reconstruction(test_df, item_or_skill=TEST_MODE)
        bt_test_df.to_csv(bt_test_path)
        if args.model == 'saint':
            datamodule = DataModule(saint_config, overwrite_test_df=bt_test_df)
            trainer = pl.Trainer(auto_select_gpus=True, callbacks=[], max_steps=0)
            bt_test_preds = predict_saint(saint_model=model, dataloader=datamodule.test_dataloader())
            print(len(bt_test_preds))
            print(bt_test_preds)
            sub_df = bt_test_df.groupby('user_id').last()
            sub_df['model_pred'] = bt_test_preds.cpu()
        else:
            bt_test_data, _ = get_data(bt_test_df, train_split=1.0, randomize=False)
            bt_test_batch = prepare_batches(bt_test_data, 10, False)
            bt_test_preds = eval_batches(model, bt_test_batch, 'cuda')
            bt_test_df['model_pred'] = bt_test_preds
            sub_df = bt_test_df.groupby('user_id').last()
        sub_df['testpass'] = (sub_df['testpoint'] == sub_df['model_pred'].round())
        sub_df.to_csv('./bt_result_{}.csv'.format(TEST_MODE))
        print(sub_df['testpass'].describe())
        print(sub_df.loc[sub_df['testpoint'] == 0, 'testpass'].describe())
        print(sub_df.loc[sub_df['testpoint'] == 1, 'testpass'].describe())
        print(new_test_meta)
    else:
        test_data, _ = get_data(test_df, train_split=1.0, randomize=False)
        test_result = torch.Tensor(test_flip_all(model, test_data))
        # print(test_result)
        print(test_result.size())
        print("All-true perturbation result:", test_result[:, 2].sum().item())
        print("All-false perturbation result:", test_result[:, 4].sum().item())
