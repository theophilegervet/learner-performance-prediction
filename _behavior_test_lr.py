import argparse
import pandas as pd
import torch

from models.model_dkt2 import DKT2
from models.model_sakt2 import SAKT

from train_dkt2 import get_data

from testcase_template import *
from utils import *


# def wrap_input(input):
#     return (torch.stack((x,)) for x in input)


# def test_flip_all(model, data):
#     with torch.no_grad():
#         test_results = []
#         for single_data in data:
#             (
#                 item_inputs,
#                 skill_inputs,
#                 label_inputs,
#                 item_ids,
#                 skill_ids,
#                 labels,
#             ) = single_data
#             orig_input = (item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
#             orig_output = model(*(wrap_input(orig_input)))
#             orig_output = torch.sigmoid(orig_output)
#             orig_output = orig_output[0][-1]
#             perturb_input, pass_range_true = generate_test_case(
#                 orig_input, orig_output, perturb_flip_all, (1,), pass_increase
#             )
#             perturb_output_true = model(*(wrap_input(perturb_input)))
#             perturb_output_true = torch.sigmoid(perturb_output_true)
#             perturb_output_true = perturb_output_true[0][-1]
#             perturb_input, pass_range_false = generate_test_case(
#                 orig_input, orig_output, perturb_flip_all, (0,), pass_decrease
#             )
#             perturb_output_false = model(*(wrap_input(perturb_input)))
#             perturb_output_false = torch.sigmoid(perturb_output_false)
#             perturb_output_false = perturb_output_false[0][-1]
#             test_results.append(
#                 [
#                     orig_output.item(),
#                     perturb_output_true.item(),
#                     float_in_range(perturb_output_true, pass_range_true).item(),
#                     perturb_output_false.item(),
#                     float_in_range(perturb_output_false, pass_range_false).item(),
#                 ]
#             )
#     return test_results


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioral Testing")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str, choices=["lr", "dkt", "sakt"])
    parser.add_argument("--test_type", nargs="+", default=[])
    parser.add_argument("--load_dir", type=str)
    parser.add_argument("--filename", type=str)
    args = parser.parse_args()

    saver = Saver(args.load_dir, args.filename)
    model = saver.load().to(torch.device("cpu"))
    model.eval()

    # testing one sample data: change first interaction
    test_df = pd.read_csv(
        os.path.join("data", args.dataset, "preprocessed_data_test.csv"), sep="\t"
    )

    test_data, _ = get_data(test_df, train_split=1.0, randomize=False)

    test_result = torch.Tensor(test_flip_all(model, test_data[:1000]))

    # print(test_result)
    print(test_result.size())
    print("All-true perturbation result:", test_result[:, 2].sum().item())
    print("All-false perturbation result:", test_result[:, 4].sum().item())
