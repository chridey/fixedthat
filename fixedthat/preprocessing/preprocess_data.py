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

def iter_data(infile, opt, lookup, predictions=None, verbose=True):
    totals = collections.Counter()    
    with open(infile) as f:
        for i,line in enumerate(f):
            if verbose and i % 10000 == 0:
                print(i)
                print(totals)
                
            j = json.loads(line)
            if j['ftfy_index'] is None:
                print(j['ftfy_index'])
                continue
            
            if opt.holdout and int(j['created']) < opt.time_partition:
                totals['time_filtered'] += 1
                continue
            elif not opt.holdout and int(j['created']) >= opt.time_partition:
                totals['time_filtered'] += 1
                continue

            key = (j['created'], j['parent_created'], j['parent_id'], j['parent_name'], j['name'])
            j.update(lookup[key])

            ftfy = ut.get_ftfy(j)
            parent = ut.get_parent_window(j, True)
            
            path = j['full_path']
            best_range = ut.get_adjusted_range(j)            
            copy = tut.path2copy(path, parent, best_range, ftfy)

            if j['ftfy_range'][0] - j['ftfy_range'][1] == 0:
                totals['delete'] += 1
            else:
                if opt.use_model and predictions is not None and not predictions[i]:
                    totals['model'] += 1
                    continue

                if len(parent) > opt.max_parent_length:
                    totals['parent_max_length'] += 1
                    continue

                if len(ftfy) > max_ftfy_length:
                    totals['ftfy_max_length'] += 1
                    continue

                if len(parent) < opt.min_parent_length:
                    totals['parent_min_length'] += 1
                    continue

                if len(ftfy) < opt.min_ftfy_length:
                    totals['ftfy_min_length'] += 1
                    continue

                if 1. * len(parent) / len(ftfy) > opt.ratio:
                    totals['ratio'] += 1
                    continue

                if j['max_char_edit'] <= opt.max_char_edit:
                    totals['max_char_edit'] += 1
                    continue

                if j['max_overlap'] >= opt.max_overlap:
                    totals['max_overlap'] += 1
                    continue

                if not opt.include_stop and (j['ftfy_index'] is None or ut.has_all_stop_words(j['ftfy_metadata'], j['ftfy_index'], j['ftfy_range'])):
                    totals['include_stop'] += 1
                    continue

                if j['non_ascii'] >= opt.non_ascii:
                    totals['non_ascii'] += 1
                    continue

            if not opt.include_json:
                del j['body']
                del j['title']
                del j['parent_body']
                j['title_metadata'] = [{'words':i['words']} for i in j['title_metadata']]
                j['ftfy_metadata'] = [{'words':i['words']} for i in j['ftfy_metadata']]
                j['parent_metadata'] = [{'words':i['words']} for i in j['parent_metadata']]
                            
            yield j, ftfy, parent, copy

parser = argparse.ArgumentParser()
parser.add_argument('infile', action='store', help='Path to data')
parser.add_argument('statsfile', action='store', help='Path to data')
parser.add_argument('pathfile', action='store', help='Path to data')
parser.add_argument('outfile', action='store', help='Path to data')
parser.add_argument('--max_parent_length', default=50, type=int)
parser.add_argument('--max_ftfy_length', default=50, type=int)
parser.add_argument('--min_parent_length', default=1, type=int)
parser.add_argument('--min_ftfy_length', default=1, type=int)
parser.add_argument('--ratio', default=9, type=int)
parser.add_argument('--max_char_edit', default=0.25, type=int)
parser.add_argument('--max_overlap', default=0.9, type=int)
parser.add_argument('--non_ascii', default=0.1, type=int)
parser.add_argument('--include_stop', action='store_true')
parser.add_argument('--add_subreddit', action='store_true')
parser.add_argument('--output_path', action='store_true')
parser.add_argument('--add_copy', action='store_true')
parser.add_argument('--holdout', action='store_true')
parser.add_argument('--time_partition', type=int, default=1508457600)
parser.add_argument('--random', type=int, default=0, help='only select a random number of examples')
parser.add_argument('--include_json', action='store_true')
parser.add_argument('--use_model', action='store_true')

opt = parser.parse_args()
print(opt)

max_ftfy_length = opt.max_ftfy_length
if opt.add_subreddit:
    max_ftfy_length += 1

opt.outfile = sys.argv[4]

lookup = {}
with open(opt.statsfile) as f:
    for line in f:
        key, value = json.loads(line)
        lookup[tuple(key)] = value

with open(opt.pathfile) as f:
    for line in f:
        key, value = json.loads(line)
        lookup[tuple(key)].update(value)

predictions = None
iterator = iter_data(opt.infile, opt, lookup, predictions)
if opt.use_model:
    df = []
    for j, ftfy, parent, copy in iter_data(opt.infile, opt, lookup):
        j.update(dict(ftfy=ftfy, parent=parent, label=None, copy=copy))
        df.append(j)
    df = pd.DataFrame(df)
    print(df.shape)
    word_vectors = dq.load_word_vectors(True)
    model = dq.load_model(True)
    print('getting features...')
    features = dq.get_features(df, word_vectors, filter_threshold=False)
    print(features.shape)
    print('making predictions...')
    predictions = dq.make_predictions(features, model, filter_features=False)
    print(predictions.shape)

    def iterator():
        for idx, row in df.iterrows():
            if predictions[idx]:
                yield row, row['ftfy'], row['parent'], row['copy']

    iterator = iterator()
#TODO: make sure sie of predictions is same
    
parents = []
parent_copies = []
ftfys = []
full_json = []

for idx, (j, ftfy, parent, copy) in enumerate(iterator):
    if idx % 10000 == 0:
        print(idx)
    if opt.add_subreddit:
        ftfy.insert(0, '__2__' + j['subreddit'] + '__')
        #parent.insert(0, '__2__' + j['subreddit'] + '__')

    if opt.add_copy:
        parent_copies.append(copy)
        #path = j['full_path']
        #best_range = ut.get_adjusted_range(j)            
        #parent_copy = tut.path2copy(path, parent, best_range, ftfy)
        #parent_copies.append(parent_copy)            

    if opt.output_path:
        path = j['full_path']
        best_range = ut.get_adjusted_range(j)
        try:
            transitions = tut.path2transitions(path, parent, best_range, ftfy)
            reconstructed_ftfy = tut.transitions2ftfy(transitions, parent)
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
            #if we cant match this, shift everything from the source and generate everything from the target
            ftfy = ['SHIFT'] * len(parent) + map(lambda x:x.lower(), ftfy) + ['REDUCE']
            print(ftfy)
            #continue

    parents.append(parent)
    ftfys.append(ftfy)
    if opt.include_json:
        full_json.append(j)
            
if opt.random:
    choices = set(np.random.choice(len(parents), opt.random, False))
else:
    choices = set(range(len(parents)))
            
with open(opt.outfile + '.tsv', 'w') as f:
    assert(len(ftfys) == len(parents))
    for i in range(len(ftfys)):
        if i not in choices:
            continue
        
        ftfy = ' '.join(ftfys[i]).encode('utf-8').replace("\t", "").replace("\r", "")
        if not opt.output_path:
            ftfy = ftfy.lower()
            
        #if opt.add_subreddit:
        #    ftfy.replace('u__subreddit__', '')

        parent = ' '.join(parents[i]).encode('utf-8').lower().replace("\t", "").replace("\r", "").replace("\xc2\xa0", '').replace('\x1e', '')
        #if opt.add_subreddit:
        #    parent.replace('u__subreddit__', '')
            
        if len(six.text_type(parent, encoding='utf-8').split()) != len(parent_copies[i]):
            continue

        #if opt.add_subreddit:
        #    ftfy = [ftfy[0], ftfy[1:]]
        #else:
        ftfy = [ftfy]
        
        if opt.add_copy:
            parent_copy = ' '.join(map(str, parent_copies[i]))
            output = [parent] + ftfy + [parent_copy]
        else:
            output = [parent] + ftfy

        if opt.holdout:
            output = [str(i)] + output

        if opt.include_json:
            output.append(json.dumps(full_json[i]))
                                
        print("\t".join(output), file=f)

        #TODO: replace ftfy, replace 'u / USER' with u__username__ and 'r / SUBREDDIT' with u__subreddit__SUBREDDIT
        #remove parents who's entire comment is a url
        
#1105609
