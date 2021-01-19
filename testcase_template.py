import torch
import pandas as pd
from copy import deepcopy


def df_perturbation(orig_df, perturb_func, pf_args):
    """
    Generates perturbed pandas dataframe object.

    Arguments:
        orig_df: original pandas dataframe object
        perturb_func: perturbation function (ex. replace, add, ...)
        pf_args: additional arguments for perturb_func
    """
    new_df_list = []
    for user_id, user_key_df in orig_df.groupby(["user_id"]):
        new_df = perturb_func(user_key_df, *pf_args)
        new_df_list.append(new_df)
    new_data = pd.concat(new_df_list, axis=0).reset_index(drop=True)
    return new_data


def perturb_review_step(orig_df):
    correct_df = orig_df.loc[orig_df["correct"] == 1]
    incorrect_df = orig_df.loc[orig_df["correct"] == 0]
    review_df = deepcopy(incorrect_df)
    review_df["correct"] = 1
    orig_df.append(review_df)
    return orig_df


def generate_test_case(
    orig_input, orig_output, perturb_func, pf_args, pass_condition, pc_args=()
):
    """
    Generates a test case with given input and output.

    Arguments:
        orig_input, orig_output : original input sequence and model output
        perturb_func : perturbation function (ex. replace, add, ...)
        pass_condition : desired range of new output as a tuple (min, max)
        pf_args, pc_args : additional arguments for perturb_func and pass_condition
    """
    return perturb_func(orig_input, *pf_args), pass_condition(orig_output, *pc_args)


def pass_invariant(orig_output, epsilon=0.1):
    return orig_output - epsilon, orig_output + epsilon


def pass_increase(orig_output, maximum_output=1):
    return orig_output, maximum_output


def pass_decrease(orig_output, minimum_output=0):
    return minimum_output, orig_output


def float_in_range(output, pass_range):
    return pass_range[0] <= output <= pass_range[1]


def perturb_flip(orig_input, replace_index):
    item_inputs, skill_inputs, label_inputs, item_ids, skill_ids = orig_input
    label_inputs[replace_index] = 1 - label_inputs[replace_index]
    return item_inputs, skill_inputs, label_inputs, item_ids, skill_ids


def perturb_flip_all(orig_input, replace_value):
    item_inputs, skill_inputs, label_inputs, item_ids, skill_ids = orig_input
    label_inputs = torch.ones(label_inputs.size()) * replace_value
    return item_inputs, skill_inputs, label_inputs, item_ids, skill_ids
