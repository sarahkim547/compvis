import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('ar', nargs=1)
args = parser.parse_args()

ar = np.load(args.ar)[0]
print(dict(zip(*np.unique(ar, return_counts=True))))
