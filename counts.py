import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('ar', nargs=1)
args = parser.parse_args()

ar = np.load(args.ar[0])
print(len(ar))
counts = dict(zip(*np.unique(ar, return_counts=True)))
print(counts)
s = sum(counts.values())
fracs = {i: c/s for i, c in counts.items()}
print(fracs)
