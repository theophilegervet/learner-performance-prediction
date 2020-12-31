import argparse
import pandas as pd
import torch

from models.model_dkt2 import DKT2
from models.model_sakt2 import SAKT

from train_dkt2 import get_data

from testcase_template import *
from utils import *


def test_flip_decrease():
    """
    Randomly flip response to correct -> incorrect.
    Expected behavior: latest probability should not increase
    """
    pass


def test_flip_increase():
    """
    Randomly flip response to incorrect -> correct.
    Expected behavior: latest probability should not decrease
    """
    pass


def wrap_input(input):
    return (torch.stack((x,)) for x in input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Behavioral Testing')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str, choices=['lr', 'dkt', 'sakt'])
    parser.add_argument('--test_type', nargs='+', default=[])
    parser.add_argument('--load_dir', type=str)
    parser.add_argument('--filename', type=str)
    args = parser.parse_args()

    saver = Saver(args.load_dir, args.filename)
    model = saver.load().to(torch.device("cpu"))
    model.eval()

    # testing one sample data: change first interaction
    with torch.no_grad():
        test_df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data_test.csv'), sep="\t")

        test_data, _ = get_data(test_df, train_split=1.0, randomize=False)
        single_data = test_data[0]
        item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels = single_data
        orig_input = (item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
        orig_output = model(*(wrap_input(orig_input)))
        orig_output = torch.sigmoid(orig_output)
        orig_output = orig_output[0][-1]
        print(orig_output)
        print(label_inputs[1])
        perturb_input, pass_range = generate_test_case(orig_input, orig_output, perturb_flip, (1,), pass_increase)
        perturb_output = model(*(wrap_input(perturb_input)))
        perturb_output = torch.sigmoid(perturb_output)
        perturb_output = perturb_output[0][-1]
        print(perturb_output)
        print(float_in_range(perturb_output, pass_range))
