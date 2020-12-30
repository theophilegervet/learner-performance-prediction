import argparse
import pandas as pd

from models.model_dkt2 import DKT2
from models.model_sakt2 import SAKT

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Behavioral Testing')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str, choices=['lr', 'dkt', 'sakt'])
    parser.add_argument('--test_type', nargs='+', default=[])
    parser.add_argument('--load_dir', type=str)
    parser.add_argument('--filename', type=str)
    args = parser.parse_args()

    saver = Saver(args.load_dir, args.filename)
    model = saver.load()
    model.eval()


    pass