import argparse
import pandas as pd
import torch

from models.model_dkt2 import DKT2
from models.model_sakt2 import SAKT

from train_dkt2 import get_data, prepare_batches, eval_batches
from train_saint import SAINT, DataModule, predict_saint

from bt_case_perturbation import (
    df_perturbation,
    perturb_add_last_random
)
from bt_case_reconstruction import test_seq_reconstruction
from bt_case_repetition import test_repeated_feed
from utils import *
import pytorch_lightning as pl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioral Testing")
    parser.add_argument("--dataset", type=str, default="spanish")
    parser.add_argument("--model", type=str, \
        choices=["lr", "dkt", "sakt", "saint"], default="saint")
    parser.add_argument("--test_type", type=str, default="reconstruction")
    parser.add_argument("--load_dir", type=str, default="./save/")
    parser.add_argument("--filename", type=str, default="spanish")
    parser.add_argument("--gpu", type=str, default="0,1")
    parser.add_argument("--diff_threshold", type=float, default=0.05)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 1. LOAD MODEL - SAINT OR {DKT, BESTLR, SAKT}
    if args.model == 'saint':
        import pickle
        checkpoint_path = f'./save/{args.model}/' + args.filename + '.ckpt'
        with open(checkpoint_path.replace('.ckpt', '_config.pkl'), 'rb') as file:
            model_config = argparse.Namespace(**pickle.load(file))
        model = SAINT.load_from_checkpoint(checkpoint_path, config=model_config\
            ).to(torch.device("cuda"))
        model.eval()
    else:
        saver = Saver(args.load_dir + f'/{args.model}/', args.filename)
        model = saver.load().to(torch.device("cuda"))
        model.eval()
        model_config = argparse.Namespace(**{})

    test_df = pd.read_csv(
        os.path.join("data", args.dataset, "preprocessed_data_test.csv"), sep="\t"
    )

    # 2. GENERATE TEST DATA.
    last_one_only = False
    if args.test_type == 'reconstruction':
        bt_test_df, test_info = test_seq_reconstruction(test_df, item_or_skill='item')
        last_one_only = True
    elif args.test_type == 'repetition':
        bt_test_df, test_info = test_repeated_feed(test_df, item_or_skill='item')
    elif args.test_type == 'add_last':
        bt_test_df, test_info = df_perturbation(test_df, perturb_add_last_random)
    elif args.test_type == 'deletion':
        raise NotImplementedError("Not implemented test_type")
    elif args.test_type == 'replacement':
        raise NotImplementedError("Not implemented test_type")
    else:
        raise NotImplementedError("Not implemented test_type")

    # 3. FEED TEST DATA.
    # In: bt_test_df
    # Out: bt_test_df with model prediction.
    # TODO: Functionize
    bt_test_path = os.path.join("data", args.dataset, "bt_{}.csv".format(args.test_type))
    original_test_df = bt_test_df.copy()
    original_test_df.to_csv(bt_test_path)
    if args.model == 'saint':
        datamodule = DataModule(model_config, overwrite_test_df=bt_test_df, last_one_only=last_one_only)
        trainer = pl.Trainer(auto_select_gpus=True, callbacks=[], max_steps=0)
        bt_test_preds = predict_saint(saint_model=model, dataloader=datamodule.test_dataloader())
        if last_one_only:
            bt_test_df = bt_test_df.groupby('user_id').last()
        bt_test_df['model_pred'] = bt_test_preds.cpu()
    else:
        bt_test_data, _ = get_data(bt_test_df, train_split=1.0, randomize=False)
        bt_test_batch = prepare_batches(bt_test_data, 10, False)
        bt_test_preds = eval_batches(model, bt_test_batch, 'cuda')
        bt_test_df['model_pred'] = bt_test_preds
        if last_one_only:
            bt_test_df = bt_test_df.groupby('user_id').last()

    # 4. CHECK PASS CONDITION AND RUN CASE-SPECIFIC ANALYSIS.
    # In: bt_test_df
    # Out: result_df (with testpass column), groupby_key
    # TODO: Functionize in separate bt_case_{}.py files.
    if args.test_type in {'reconstruction', 'repetition'}:
        bt_test_df['testpass'] = (bt_test_df['testpoint'] == bt_test_df['model_pred'].round())
        groupby_key = ['all', 'testpoint']
        result_df = bt_test_df
    elif args.test_type == 'add_last':
        user_group_df = bt_test_df.groupby('orig_user_id')
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
        result_df = user_group_df.loc[user_group_df['is_perturbed'] != 0]
        groupby_key = ['all', 'is_pertubred']
    elif args.test_type == 'deletion':
        raise NotImplementedError("Not implemented test_type")
    elif args.test_type == 'replacement':
        raise NotImplementedError("Not implemented test_type")
    else:
        raise NotImplementedError("Not implemented test_type")

    # 5. GET COMMON TEST CASE STAT.
    result_dict = {}
    eval_col = 'testpass'
    result_df['all'] = 'all'
    for group_key in groupby_key:
        result_dict[group_key] = result_df.groupby(group_key)[eval_col].describe()
    metric_df = pd.concat([y for _, y in result_dict.items()], axis=0, keys=result_dict.keys())
    print(metric_df)
    result_df.to_csv(f'./results/{args.dataset}_{args.test_type}_{args.model}.csv')
