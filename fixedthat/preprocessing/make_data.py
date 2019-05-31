from __future__ import print_function

import collections
import sys
import datetime
import json
import argparse

#import pandas as pd

import six
import numpy as np
import pandas as pd

import ftfy_utils as ut
import transition_utils as tut
import data_quality as dq

import wikipedia_utils as wut

infile = sys.argv[1]

splits = {'train': [], 'dev': [], 'blind': []}
blind_partition = 1501560000
dev_partition = 1493611200
subset_categories = {'Entertainment', 'Gaming', 'Lifestyle', 'Locations', 'News and Politics', 'Sports', 'Technology'}
categories = wut.get_categories()
title_lookup,_ = wut.build_link_lookup(sys.argv[2], sys.argv[3], None)
with open(infile) as f:
    for ix,line in enumerate(f):
        if ix % 10000 == 0:
            print(ix)                
        j = json.loads(line)

        if not len(zip(*j['pmi'])):
            continue
        if not len(j['links']):
            continue
        if j['subreddit'].lower() not in categories or not len(categories[j['subreddit'].lower()]) or categories[j['subreddit'].lower()][0] not in subset_categories:
            continue
    
        parent = ut.get_parent_window(j, True)
        parent_pos = ut.get_parent_window(j, True, key='pos')
        parent_lemmas = ut.get_parent_window(j, True, key='lemmas')
        ftfy = ut.get_ftfy(j)
        ftfy_pos = ut.get_ftfy(j, key='pos')
        ftfy_lemmas = ut.get_ftfy(j, key='lemmas')
        
        parent = wut.segment_collocations(title_lookup, parent, parent_lemmas, parent_pos, collocations_only=False)
        ftfy = wut.segment_collocations(title_lookup, ftfy, ftfy_lemmas, ftfy_pos, collocations_only=False)
        
        ftfy = ' '.join(ftfy).encode('utf-8').replace("\t", "").replace("\r", "").lower()

        parent = ' '.join(parent).encode('utf-8').lower().replace("\t", "").replace("\r", "").replace("\xc2\xa0", '').replace('\x1e', '')

        output = [parent, ftfy, ' '.join(zip(*j['pmi'])[0]).encode('utf-8')]

        split = 'train'
        if int(j['created']) > blind_partition:
            split = 'blind'
            output = [str(len(splits[split]))] + output            
        elif int(j['created']) > dev_partition:
            split = 'dev'
                                    
        splits[split].append(output)
        
for key in splits:
    with open(infile + '.' + key, 'w') as f:
        for output in splits[key]:
            try:
                print("\t".join(output), file=f)
            except Exception as e:
                print(e)
        
