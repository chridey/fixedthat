from __future__ import print_function

import gzip
import collections
import sys
import datetime
import json
import argparse
import csv

import numpy as np

import ftfy_utils as ut
import wikipedia_utils as wut

infile = sys.argv[1]
title_lookup, link_lookup = wut.build_link_lookup(sys.argv[2], sys.argv[3], None)

parent_side_not_ftfy_side = collections.Counter()
parent_side = collections.Counter()
ftfy_side = collections.Counter()
total = 0.

subset_categories = {'Entertainment', 'Gaming', 'Lifestyle', 'Locations', 'News and Politics', 'Sports', 'Technology'}
categories = wut.get_categories()

with open(infile) as f:
    for ix,line in enumerate(f):
        
        j = json.loads(line)
        if not len(j['links']):
            continue            
        if j['subreddit'].lower() not in categories or not len(categories[j['subreddit'].lower()]) or categories[j['subreddit'].lower()][0] not in subset_categories:
            continue
        
        ftfy = ut.get_ftfy(j)
        ftfy_pos = ut.get_ftfy(j, key='pos')
        ftfy_lemmas = ut.get_ftfy(j, key='lemmas')

        parent = ut.get_parent_window(j, True)
        parent_pos = ut.get_parent_window(j, True, key='pos')
        parent_lemmas = ut.get_parent_window(j, True, key='lemmas')
        
        parent_sc = set(wut.segment_collocations(title_lookup, parent, parent_lemmas, parent_pos, collocations_only=True))
        ftfy_sc = set(wut.segment_collocations(title_lookup, ftfy, ftfy_lemmas, ftfy_pos, collocations_only=True)) - parent_sc
        
        for sc in parent_sc:
            parent_side[sc] += 1
            if sc not in ftfy_sc:
                parent_side_not_ftfy_side[sc] += 1
        for sc in ftfy_sc:
            ftfy_side[sc] += 1
        total += 1

pmi_side = {}
for sc in parent_side_not_ftfy_side:
    pmi_side[sc] = (parent_side_not_ftfy_side[sc]/total)/((parent_side[sc]/total)*((total-ftfy_side[sc])/total))

outfile = infile + '.pmi_side'
with open(outfile, 'w') as f:
    for key,value in sorted(pmi_side.items(), key=lambda x:x[1]):
        print('\t'.join([key, str(value)]), file=f)
    
