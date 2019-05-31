'''
script for aligning the parent comment and FTFY using edit distance
usage: parse_ftfy.py <infile> <info_outfile> <force>

writes to stdout
info_outfile write statistics for how often the FTFY is a quote or edit
force is whether the infile is already tokenized and aligned and we are just adding some metadata
'''

from __future__ import print_function

import sys
import csv
import collections
import json

from ftfy_utils import *

infile = sys.argv[1]
info = sys.argv[2]
force = int(sys.argv[3])

quote = set()
edit = set()
other = set()
subreddit_counts = collections.Counter()

with open(infile) as f:
    reader = f
    for index,line in enumerate(reader):

        try:
            row = json.loads(line)
        except ValueError:
            print('unable to load json from {}'.format(line), file=sys.stderr)
            continue
        
        if row['body'].strip() in ('', '[deleted]', '[removed]') or (row['parent_body'].strip() in ('', '[deleted]', '[removed]') and row['title'].strip() in ('', '[deleted]', '[removed]')):
            continue
        subreddit_counts[row['subreddit']] += 1

        if row['body'].startswith('&gt;'):
            quote.add(index)                
        if '~~' in row['body']:
            edit.add(index)

        other.add(index)
        
        if force or 'ftfy_metadata' not in row:
            ftfy = tokenize(row['body'])
            ftfy_index = get_ftfy_index(ftfy)

            parent = tokenize(row['parent_body'])
            parent_min, parent_range = get_sentence_alignment(ftfy, ftfy_index, parent)
            title = tokenize(row['title'])
            title_min, title_range = get_sentence_alignment(ftfy, ftfy_index, title)

            if parent_min < title_min:
                best = 'parent'
                best_tokenized = parent
                best_range = parent_range
                best_score = parent_min
            else:
                best = 'title'
                best_tokenized = title
                best_range = title_range
                best_score = title_min
        else:
            best = row['best']
            best_range = row['best_range']
            best_score = row['best_score']
            ftfy_index = row['ftfy_index']
            ftfy = row['ftfy_metadata']                
            parent = row['parent_metadata']                
            title = row['title_metadata']                                
            best_tokenized = row['{}_metadata'.format(best)]

        ftfy_range, parent_range, labels = get_all_sentence_labels(ftfy, ftfy_index,
                                                                   best_tokenized,
                                                                   best_range)
        if parent_range is not None:
            ftfy_sentence = join_sentences(get_search_string(ftfy, ftfy_index),
                                           True)[ftfy_range[0]:ftfy_range[1]]
            parent_sentence = join_sentences(best_tokenized)[parent_range[0]:parent_range[1]]
            num_non_ascii = get_num_non_ascii(ftfy_sentence)
            max_overlap = get_max_overlap(ftfy_sentence, parent_sentence)
            metadata = {i:row[i] for i in row}
            metadata.update(dict(best=best,
                                 best_range=best_range,
                                 best_score=best_score,
                                 ftfy_index=ftfy_index,
                                 ftfy_metadata=ftfy,
                                 parent_metadata=parent,
                                 title_metadata=title,
                                 labels=labels,
                                 ftfy_range=ftfy_range,
                                 parent_range=parent_range,
                                 max_overlap=max_overlap,
                                 num_non_ascii=num_non_ascii))
            print(json.dumps(metadata))
                
with open(info, 'w') as f:                
    print(len(other), file=f)
    print(len(quote), file=f)
    print(len(edit), file=f)
    print(len(quote & edit), file=f)        

    for pair in sorted(subreddit_counts.items(), key=lambda x:x[1]):
        print(pair, file=f)
    
