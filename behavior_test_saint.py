import argparse
import pandas as pd
import torch
import pytorch_lightning as pl

from train_saint import get_data, InteractionDataset, SAINT

from testcase_template import *
from utils import *
from tqdm import tqdm

import copy


def wrap_input(input):
    return (torch.stack((x,)) for x in input)


def perturb_flip_all_saint(orig_input, replace_value):
    perturb_input = copy.deepcopy(orig_input)
    perturb_input["correct"] = [replace_value] * len(perturb_input["correct"])
    return perturb_input


class NewDataModule(pl.LightningDataModule):
    def __init__(
        self, data, seq_len=100, train_batch=128, test_batch=128, num_workers=0
    ):
        super().__init__()
        self.data = data
        train_data = InteractionDataset(self.data["train"], seq_len=seq_len,)
        val_data = InteractionDataset(self.data["val"], seq_len=seq_len,)
        test_data = InteractionDataset(self.data["test"], seq_len=seq_len, stride=1, is_test=True)
        self.train_gen = torch.utils.data.DataLoader(
            dataset=train_data,
            shuffle=True,
            batch_size=train_batch,
            num_workers=num_workers,
        )
        self.val_gen = torch.utils.data.DataLoader(
            dataset=val_data,
            shuffle=False,
            batch_size=test_batch,
            num_workers=num_workers,
        )
        self.test_gen = torch.utils.data.DataLoader(
            dataset=test_data,
            shuffle=False,
            batch_size=test_batch,
            num_workers=num_workers,
        )

    def train_dataloader(self):
        return self.train_gen

    def test_dataloader(self):
        return self.test_gen

    def val_dataloader(self):
        return self.val_gen


def test_flip_all_saint(model, dataset, seq_len=100, gpu=0, device="cpu"):
    data = get_data(dataset)
    data_true = copy.deepcopy(data)
    data_false = copy.deepcopy(data)
    for uid in data_true["test"]:
        data_true["test"][uid] = perturb_flip_all_saint(data_true["test"][uid], 1)
        data_false["test"][uid] = perturb_flip_all_saint(data_false["test"][uid], 0)
    datamodule = NewDataModule(data)
    datamodule_true = NewDataModule(data_true)
    datamodule_false = NewDataModule(data_false)
    trainer = pl.Trainer(gpus=gpu, callbacks=[],)
    results = []
    for dm in [datamodule, datamodule_true, datamodule_false]:
        loader = dm.test_dataloader()
        preds = []
        for batch in tqdm(loader):
            test_res = model.compute_all_losses(batch)
            pred = test_res["pred"]
            infer_mask = batch["infer_mask"]
            nonzeros = torch.nonzero(infer_mask, as_tuple=True)
            pred = pred[nonzeros].sigmoid()
            preds.append(pred)
        preds = torch.cat(preds, dim=0).view(-1)
        results.append(preds)

    return results


def str2bool(val):
    if val.lower() in ("yes", "true", "t", "y", "1"):
        ret = True
    elif val.lower() in ("no", "false", "f", "n", "0"):
        ret = False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioral Testing")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str, choices=["lr", "dkt", "sakt", "saint"])
    parser.add_argument("--test_type", nargs="+", default=[])
    parser.add_argument("--load_dir", type=str, default="weight/saint")
    parser.add_argument("--filename", type=str, default="ednet_small.ckpt")
    parser.add_argument("--gpu", type=str)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--val_check_steps", type=int, default=100)
    parser.add_argument("--random_seed", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=50000)
    parser.add_argument("--train_batch", type=int, default=128)
    parser.add_argument("--test_batch", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--layer_count", type=int, default=2)
    parser.add_argument("--head_count", type=int, default=8)
    parser.add_argument("--embed_sum", type=str2bool, default=False)
    parser.add_argument("--warmup_step", type=int, default=4000)
    parser.add_argument("--dim_model", type=int, default=256)
    parser.add_argument("--dim_ff", type=int, default=1024)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--project", type=str)
    parser.add_argument("--name", type=str)
    args = parser.parse_args()

    if args.dataset in ["ednet", "ednet_medium"]:
        args.num_item = 14000
        args.num_skill = 300
    else:
        full_df = pd.read_csv(
            os.path.join("data", args.dataset, "preprocessed_data.csv"), sep="\t"
        )
        args.num_item = int(full_df["item_id"].max() + 1)
        args.num_skill = int(full_df["skill_id"].max() + 1)

    model = SAINT.load_from_checkpoint(
        os.path.join(args.load_dir, args.filename), config=args
    )

    result, result_true, result_false = test_flip_all_saint(
        model, args.dataset, gpu=args.gpu, device=args.device
    )
    num_inter = result.size()[0]
    true_cnt, false_cnt = 0, 0
    for i in range(num_inter):
        if result[i].item() <= result_true[i].item():
            true_cnt += 1
        if result[i].item() >= result_false[i].item():
            false_cnt += 1
    with open(os.path.join(args.load_dir, "result.txt"), "w") as f:
        f.write(str(num_inter)+" "+str(true_cnt)+" "+str(false_cnt))
