from __future__ import print_function

import collections
import sys

import json
import pandas as pd

import ftfy_utils as ut

max_length = 25
output_path = True
add_subreddit = False

if add_subreddit:
    max_length += 1

infile = sys.argv[1]
stats = sys.argv[2]
path = sys.argv[3]
outfile = sys.argv[4]

lookup = {}
with open(stats) as f:
    for line in f:
        key, value = json.loads(line)
        lookup[tuple(key)] = value

with open(path) as f:
    for line in f:
        key, value = json.loads(line)
        lookup[tuple(key)].update(value)

parents = []
ftfys = []
totals = collections.Counter()
with open(infile) as f:
    for i,line in enumerate(f):
        if i % 10000 == 0:
            print(i)
            print(totals)
        j = json.loads(line)
        key = (j['created'], j['parent_created'], j['parent_id'], j['parent_name'], j['name'])
        j.update(lookup[key])

        ftfy = ut.get_ftfy(j)
        parent = ut.get_parent_window(j, True)
        if add_subreddit:
            ftfy.insert(0, j['subreddit'])
            parent.insert(0, j['subreddit'])
                    
        if j['ftfy_range'][0] - j['ftfy_range'][1] == 0:
            totals['delete'] += 1
        else:
            if len(parent) > max_length:
                totals['best_length'] += 1
                continue
            
            if len(ftfy) > max_length:
                totals['ftfy_length'] += 1
                continue
        
            if j['max_char_edit'] <= 0.25:
                totals['max_char_edit'] += 1
                continue
            
            if j['max_overlap'] >= 0.9:
                totals['max_overlap'] += 1
                continue

            if ut.has_all_stop_words(j['ftfy_metadata'], j['ftfy_index'], j['ftfy_range']):
                totals['stop'] += 1
                continue

            if j['non_ascii'] >= 0.1:
                totals['non_ascii'] += 1
                continue
            
        if output_path:
            path = j['full_path']
            best_range = ut.get_adjusted_range(j)
            try:
                transitions = ut.path2transitions(path, parent, best_range, ftfy)
                reconstructed_ftfy = ut.transitions2ftfy(transitions, parent)
            except IndexError:
                transitions = []
                reconstructed_ftfy = []
            except AssertionError:
                totals['bad_path'] += 1
                continue
            if reconstructed_ftfy == map(lambda x:x.lower(), ftfy):
                ftfy = transitions
            else:
                totals['bad_transitions'] += 1
                continue
            
        parents.append(parent)
        ftfys.append(ftfy)
        
with open(outfile + '.ftfy', 'w') as f:
    for ftfy in ftfys:
        ftfy = ' '.join(ftfy).encode('utf-8')
        if not output_path:
            ftfy = ftfy.lower()
            
        if add_subreddit:
            ftfy.replace('u__subreddit__', '')
        print(ftfy, file=f)

with open(outfile + '.parent', 'w') as f:
    for parent in parents:
        parent = ' '.join(parent).encode('utf-8').lower()
        if add_subreddit:
            parent.replace('u__subreddit__', '')
        print(parent, file=f)
