import argparse
import pandas as pd
import torch

from models.model_dkt2 import DKT2
from models.model_sakt2 import SAKT

import pickle
from train_dkt2 import get_data, prepare_batches, eval_batches
from train_saint import SAINT, DataModule, predict_saint

from bt_case_perturbation import (
    gen_perturbation,
    perturb_insertion_random, perturb_delete_random, test_perturbation,
)
from bt_case_reconstruction import gen_seq_reconstruction, test_simple
from bt_case_repetition import gen_repeated_feed
from utils import *
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, accuracy_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioral Testing")
    parser.add_argument("--dataset", type=str, default="spanish")
    parser.add_argument("--model", type=str, \
        choices=["lr", "dkt", "sakt", "saint"], default="dkt")
    parser.add_argument("--test_type", type=str, default="original")
    parser.add_argument("--load_dir", type=str, default="./save/")
    parser.add_argument("--filename", type=str, default="spanish")
    parser.add_argument("--gpu", type=str, default="0,1")
    parser.add_argument("--diff_threshold", type=float, default=0.05)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 1. LOAD DATA + PRE-TRAINED MODEL - {SAINT, DKT, BESTLR, SAKT}
    if args.model == 'saint':
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
    test_kwargs = {
        'item_or_skill': 'item', 
        'perturb_func': {
            'insertion': perturb_insertion_random,
            'deletion': perturb_delete_random
        }[args.test_type],
        'insertion_policy': 'middle'
        }
    last_one_only = {'reconstruction': True, 'repetition': False, \
        'insertion': False, 'original': False}[args.test_type]


    # 2. GENERATE TEST DATA.
    gen_funcs = {
        'reconstruction': gen_seq_reconstruction,
        'repetition': gen_repeated_feed,
        'insertion': gen_perturbation,
        'deletion': gen_perturbation,
        'original': lambda x: x
    }
    bt_test_df, test_info = gen_funcs[args.test_type](test_df, **test_kwargs)
        # bt_test_df: generated test dataset for behavioral testing
        # test_info: any meta information stored w.r.t the test case


    # 3. FEED TEST DATA.
    # In: bt_test_df
    # Out: bt_test_df with 'model_pred' column.
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
    test_funcs = {
        'reconstruction': test_simple,
        'repetition': test_simple,
        'insertion': test_perturbation,
        'deletion': test_perturbation,
        'original': lambda x: test_simple(x, testcol='correct')
    }
    result_df, groupby_key = test_funcs[args.test_type](bt_test_df)
        # result_df: bt_test_df appended with 'testpass' column or any additional test-case-specific info.
        # groupby_key: list of column names in result_df for 5's mutually exclusive subset-wise analysis.


    # 5. GET SUMMARY STAT.
    result_dict = {}
    eval_col = 'testpass'
    result_df['all'] = 'all'
    for group_key in groupby_key:
        result_dict[group_key] = result_df.groupby(group_key)[eval_col].describe()
    metric_df = pd.concat([y for _, y in result_dict.items()], axis=0, keys=result_dict.keys())
    print(metric_df)
    print("auc_test = ", roc_auc_score(result_df["correct"], result_df['model_pred']))
    result_df.to_csv(f'./results/{args.dataset}_{args.test_type}_{args.model}.csv')
