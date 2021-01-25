import torch
import pandas as pd
from copy import deepcopy
import random


def test_perturbation(bt_test_df, diff_threshold=0.05):
    bt_test_df['testpass'] = False
    user_group_df = bt_test_df.groupby('orig_user_id')
    for name, group in user_group_df:
        orig_prob = group.loc[group['is_perturbed'] == 0]['model_pred'].item()
        corr_prob = group.loc[group['is_perturbed'] == 1]['model_pred'].item()
        incorr_prob = group.loc[group['is_perturbed'] == -1]['model_pred'].item()
        if corr_prob >= orig_prob - diff_threshold:
            bt_test_df.loc[
                (bt_test_df['orig_user_id'] == name) & (bt_test_df['is_perturbed'] == 1),
                'testpass'] = True
        if incorr_prob <= orig_prob + diff_threshold:
            bt_test_df.loc[
                (bt_test_df['orig_user_id'] == name) & (bt_test_df['is_perturbed'] == -1),
                'testpass'] = True

    result_df = user_group_df.loc[user_group_df['is_perturbed'] != 0]
    groupby_key = ['all', 'is_perturbed']
    return result_df, groupby_key


def gen_perturbation(orig_df, perturb_func, **pf_args):
    """
    Generates perturbed pandas dataframe object.

    Arguments:
        orig_df: original pandas dataframe object
        perturb_func: perturbation function (ex. replace, add, ...)
        pf_args: additional arguments for perturb_func
    """
    new_df_list = []
    for user_id, user_key_df in orig_df.groupby(["user_id"]):
        new_df = perturb_func(user_key_df, **pf_args)
        new_df_list.append(new_df)
    new_data = pd.concat(new_df_list, axis=0).reset_index(drop=True)
    data_meta = {
        'num_sample': new_data['user_id'].unique().shape[0],
        'num_interaction': new_data.shape[0],
    }
    return new_data, data_meta


def perturb_review_step(orig_df):
    correct_df = orig_df.loc[orig_df["correct"] == 1]
    incorrect_df = orig_df.loc[orig_df["correct"] == 0]
    review_df = deepcopy(incorrect_df)
    review_df["correct"] = 1
    orig_df = orig_df.append(review_df).reset_index(drop=True)
    return orig_df


def perturb_add_last(orig_df, row_index, new_value):
    new_df = deepcopy(orig_df.iloc[[row_index]])
    new_df.loc[:, "correct"] = new_value
    orig_df = orig_df.append(new_df).reset_index(drop=True)
    return orig_df


def perturb_add_last_random(orig_df):
    orig_df.loc[:, 'orig_user_id'] = orig_df['user_id']
    orig_df.loc[:, 'is_perturbed'] = 0
    row_index = random.randrange(0, len(orig_df))
    corr_df = perturb_add_last(orig_df, row_index, 1)
    corr_df.loc[:, 'user_id'] = corr_df['user_id'].astype(str) + "corr"
    corr_df.loc[:, 'is_perturbed'] = 1
    incorr_df = perturb_add_last(orig_df, row_index, 0)
    incorr_df.loc[:, 'user_id'] = incorr_df['user_id'].astype(str) + "incorr"
    incorr_df.loc[:, 'is_perturbed'] = -1

    orig_df = orig_df.append(orig_df.iloc[row_index]).reset_index(drop=True)
    corr_df = corr_df.append(corr_df.iloc[row_index]).reset_index(drop=True)
    incorr_df = incorr_df.append(incorr_df.iloc[row_index]).reset_index(drop=True)

    new_df_list = [orig_df, corr_df, incorr_df]
    return pd.concat(new_df_list, axis=0).reset_index(drop=True)


def perturb_insertion(orig_df, copy_idx, insert_idx, corr_value):
    new_df = deepcopy(orig_df.iloc[[copy_idx]])
    new_df.loc[:, "correct"] = corr_value
    orig_df = orig_df.iloc[:insert_idx].append(new_df)\
        .append(orig_df.iloc[insert_idx:]).reset_index(drop=True)
    return orig_df


def perturb_insertion_random(orig_df, insert_policy=None):
    orig_df.loc[:, 'orig_user_id'] = orig_df['user_id']
    orig_df.loc[:, 'is_perturbed'] = 0
    copy_idx = random.randrange(0, len(orig_df))
    if insert_policy == "first":
        insert_idx = 0
    elif insert_policy == "middle":
        insert_idx = len(orig_df) // 2
    elif insert_policy == "last":
        insert_idx = len(orig_df) - 1
    else:
        insert_idx = random.randrange(0, len(orig_df))
    corr_df = perturb_insertion(orig_df, copy_idx, insert_idx, 1)
    corr_df.loc[:, 'user_id'] = corr_df['user_id'].astype(str) + "_corr"
    corr_df.loc[:, 'is_perturbed'] = 1
    incorr_df = perturb_insertion(orig_df, copy_idx, insert_idx, 0)
    incorr_df.loc[:, 'user_id'] = incorr_df['user_id'].astype(str) + "_incorr"
    incorr_df.loc[:, 'is_perturbed'] = -1

    new_df_list = [orig_df, corr_df, incorr_df]
    return pd.concat(new_df_list, axis=0).reset_index(drop=True)


def perturb_delete(orig_df, row_index):
    orig_df.loc[:, 'is_perturbed'] = 1 if (orig_df.iloc[row_index]['correct'] == 1) else -1
    orig_df = orig_df.iloc[:row_index].append(orig_df.iloc[row_index+1:]).reset_index(drop=True)
    return orig_df


def perturb_delete_random(orig_df, delete_policy=None):
    orig_df.loc[:, 'orig_user_id'] = orig_df['user_id']
    orig_df.loc[:, 'is_perturbed'] = 0
    if delete_policy == "first":
        row_idx = 0
    elif delete_policy == "middle":
        row_idx = len(orig_df) // 2
    elif delete_policy == "last":
        row_idx = len(orig_df) - 1
    else:
        row_idx = random.randrange(0, len(orig_df))
    row_index = random.randrange(0, len(orig_df))
    new_df = perturb_delete(orig_df, row_idx)
    new_df.loc[:, 'user_id'] = new_df['user_id'].astype(str) + "_del"
    new_df_list = [orig_df, new_df]
    return pd.concat(new_df_list, axis=0).reset_index(drop=True)

# depercated templates


# def generate_test_case(
#     orig_input, orig_output, perturb_func, pf_args, pass_condition, pc_args=()
# ):
#     """
#     Generates a test case with given input and output.
#
#     Arguments:
#         orig_input, orig_output : original input sequence and model output
#         perturb_func : perturbation function (ex. replace, add, ...)
#         pass_condition : desired range of new output as a tuple (min, max)
#         pf_args, pc_args : additional arguments for perturb_func and pass_condition
#     """
#     return perturb_func(orig_input, *pf_args), pass_condition(orig_output, *pc_args)
#
#
# def pass_invariant(orig_output, epsilon=0.1):
#     return orig_output - epsilon, orig_output + epsilon
#
#
# def pass_increase(orig_output, maximum_output=1):
#     return orig_output, maximum_output
#
#
# def pass_decrease(orig_output, minimum_output=0):
#     return minimum_output, orig_output
#
#
# def float_in_range(output, pass_range):
#     return pass_range[0] <= output <= pass_range[1]
#
#
# def perturb_flip(orig_input, replace_index):
#     item_inputs, skill_inputs, label_inputs, item_ids, skill_ids = orig_input
#     label_inputs[replace_index] = 1 - label_inputs[replace_index]
#     return item_inputs, skill_inputs, label_inputs, item_ids, skill_ids
#
#
# def perturb_flip_all(orig_input, replace_value):
#     item_inputs, skill_inputs, label_inputs, item_ids, skill_ids = orig_input
#     label_inputs = torch.ones(label_inputs.size()) * replace_value
#     return item_inputs, skill_inputs, label_inputs, item_ids, skill_ids
