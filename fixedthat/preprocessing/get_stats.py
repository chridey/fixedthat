from __future__ import print_function

import json
import pandas as pd
import sys
import collections

import ftfy_utils as ut

def get_stats(item):
    ftfy = item['ftfy_metadata']
    try:
        ftfy_index = int(item['ftfy_index'])
    except TypeError:
        return None, None, None
    
    ftfy_range = item['ftfy_range']
    parent = item['{}_metadata'.format(item['best'])]
    parent_range = item['parent_range']

    #print(ftfy_index, ftfy_range, parent_range)
    ftfy_sentence = ut.join_sentences(ut.get_search_string(ftfy, ftfy_index), True)[ftfy_range[0]:ftfy_range[1]]
    parent_sentence = ut.join_sentences(parent)[parent_range[0]:parent_range[1]]
    #print(ftfy_sentence, parent_sentence)
    non_ascii = ut.get_num_non_ascii(ftfy_sentence)
    max_overlap = ut.get_max_overlap(ftfy_sentence, parent_sentence)
    max_char_edit = ut.get_max_char_edit(ftfy_sentence, parent_sentence)

    return non_ascii, max_overlap, max_char_edit

full = sys.argv[1]

lookup = collections.defaultdict(list)
if len(sys.argv) > 2:
    stats = sys.argv[2]
    
    with open(stats) as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                print(line, file=sys.stderr)
                continue

            key = (j['name'], j['parent_id'], j['parent_name'])
            lookup[key].append(j)
            #if len(lookup[key]) > 1:
            #    print(j, file=sys.stderr)

train = []
final_lookup = {}
with open(sys.argv[1]) as f:
    for i,line in enumerate(f):
        if i % 10000 == 0:
            print(i, file=sys.stderr)

        j = json.loads(line)

        key = (j['created'], j['parent_created'], j['parent_id'], j['parent_name'], j['name'])
        if key in final_lookup:
            continue

        pre_key = (j['name'], j['parent_id'], j['parent_name'])
        if pre_key in lookup and len(lookup[pre_key]) == 1:
            j = lookup[pre_key][0]
            non_ascii, max_overlap, max_char_edit = j['non_ascii'], j['max_overlap'], j['max_char_edit']
        else:        
            non_ascii, max_overlap, max_char_edit = get_stats(j)

        value = dict(non_ascii=non_ascii, max_overlap=max_overlap, max_char_edit=max_char_edit)
        final_lookup[key] = value

        print(json.dumps([key, value]))
        
#print(json.dumps(final_lookup))
