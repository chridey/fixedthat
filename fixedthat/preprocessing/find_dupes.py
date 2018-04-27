import sys
import collections

lookup = collections.Counter()
with open(sys.argv[1]) as f:
    for line in f:
        idx, parent, ftfy, template = line.split('\t')
        lookup[parent] += 1

for i in lookup:
    if lookup[i] > 1:
        print(i, lookup[i])        
