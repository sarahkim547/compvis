import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('ar', nargs=1, required=True)
args = parser.parse_args()

ar = np.load(args.ar)
print(dict(zip(*np.unique(ar, return_counts=True))))
