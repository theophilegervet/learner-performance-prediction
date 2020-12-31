def generate_test_case(orig_input, orig_output, perturb_func, pf_args, pass_condition, pc_args=()):
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
