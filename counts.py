import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--ar', required=True)
args = parser.parse_args()

ar = np.load(args.ar)
if len(ar.shape) == 3:
    ar = np.argmax(ar, axis=2)
ar = ar.flatten()
print(len(ar))
counts = dict(zip(*np.unique(ar, return_counts=True)))
print(counts)
s = sum(counts.values())
fracs = {i: c/s for i, c in counts.items()}
print(fracs)
