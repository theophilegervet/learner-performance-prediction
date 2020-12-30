import argparse
import pandas as pd

from testcase_template import *


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Behavioral Testing')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str, choices=['lr', 'dkt', 'sakt'])
    parser.add_argument('--test_type', nargs='+', default=[])
    args = parser.parse_args()
    pass