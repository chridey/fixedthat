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
title_lookup, link_lookup = wut.build_link_lookup(sys.argv[2], sys.argv[3], sys.argv[4])
reverse_title_lookup = collections.defaultdict(list)
for key in title_lookup:
    for value in title_lookup[key]:
        reverse_title_lookup[value].append(key)

pmi = None
if len(sys.argv) > 5:
    collocations_file = sys.argv[5]
    collocation_counts = collections.defaultdict(lambda: collections.Counter())
    ftfy_counts = collections.Counter()
    parent_counts = collections.Counter()
    with open(collocations_file) as f:
        for line in f:
            collocation_pairs = json.loads(line)
            found_ftfy = set()
            found_parent = set()
            for ftfy_collocation,parent_collocation in collocation_pairs:
                if ftfy_collocation not in found_ftfy:
                    found_ftfy.add(ftfy_collocation)
                    ftfy_counts[ftfy_collocation] += 1
                if parent_collocation not in found_parent:
                    found_parent.add(parent_collocation)
                    parent_counts[parent_collocation] += 1
                collocation_counts[parent_collocation][ftfy_collocation] += 1    

    spf = 1.*sum(sum(i.values()) for i in collocation_counts.values())
    sf = 1.*sum(ftfy_counts.values())
    sp = 1.*sum(parent_counts.values())

    pmi = collections.defaultdict(dict)
    
    for p in collocation_counts:
        for f in collocation_counts[p]:
            if collocation_counts[p][f] > 1:
                pmi[p][f] = (collocation_counts[p][f]/spf)/((ftfy_counts[f]/sf)*(parent_counts[p]/sp))
                
splits = {'train': [], 'dev': [], 'blind': []}
blind_partition = 1501560000
dev_partition = 1493611200
subset_categories = {'Entertainment', 'Gaming', 'Lifestyle', 'Locations', 'News and Politics', 'Sports', 'Technology'}
categories = wut.get_categories()
with open(infile) as f:
    all_collocations = []
    for ix,line in enumerate(f):
        #if ix == 10000:
        #    break
        if ix % 1000 == 0:
            print(ix)                
        j = json.loads(line)

        if not len(j['links']):
            continue
        if j['subreddit'].lower() not in categories or not len(categories[j['subreddit'].lower()]) or categories[j['subreddit'].lower()][0] not in subset_categories:
            continue

        if pmi is not None:
            parent = ut.get_parent_window(j, True)
            parent_pos = ut.get_parent_window(j, True, key='pos')
            parent_lemmas = ut.get_parent_window(j, True, key='lemmas')

            sc = set(wut.segment_collocations(title_lookup, parent, parent_lemmas, parent_pos, collocations_only=True))
            sc_pmi = collections.defaultdict(float)                
            for segment in sc:
                for key in pmi[segment]:
                    if key in sc:
                        continue
                    sc_pmi[key] += pmi[segment][key]

            j['pmi'] = sorted(sc_pmi.items(), key=lambda x:x[1])[-100:]                    
            if len(j['pmi']) < 100:
                #first add all the direct links of the parents and randomly prune to 100

                potential_titles = []
                first_hop_links = []
                for segment in sc:
                    for index in title_lookup[segment]:
                        first_hop_links.extend(link_lookup[index])
                        for link in link_lookup[index]:
                            for title in reverse_title_lookup[link]:
                                if title in sc:
                                    continue
                                potential_titles.append(title)

                def add_titles(potential_titles, distance):
                    num_to_add = 100-len(j['pmi'])
                    potential_titles = list(set(potential_titles))
                    if len(potential_titles) >= num_to_add:
                        selections = np.random.choice(len(potential_titles), num_to_add, False)
                        titles_to_add = [potential_titles[i] for i in selections]
                    else:
                        titles_to_add = potential_titles
                    for title in titles_to_add:
                        j['pmi'].append((title, distance))
                add_titles(potential_titles, 0)
                
                #then if still not at 100, add the second hop links
                second_hop_titles = []
                if len(j['pmi']) < 100:
                    for index in first_hop_links:
                        for link in link_lookup[index]:
                            for title in reverse_title_lookup[link]:
                                if title in sc:
                                    continue
                                second_hop_titles.append(title)
                    add_titles(second_hop_titles, -1)
                
            all_collocations.append(j)
            
            continue
        
        ftfy = j['ftfy_metadata']
        ftfy_index = j['ftfy_index']
        parent = j['{}_metadata'.format(j['best'])]
        parent_range = j['best_range']
        ftfy_range, parent_range = ut.find_exact_ftfy_boundaries(ftfy, ftfy_index, parent, parent_range)

        ftfy_pos = ut.join_sentences(ut.get_search_string(ftfy, ftfy_index), True, key='pos')[ftfy_range[0]:ftfy_range[1]]
        parent_pos = ut.join_sentences(parent, key='pos')[parent_range[0]:parent_range[1]]
        ftfy_lemmas = ut.join_sentences(ut.get_search_string(ftfy, ftfy_index), True, key='lemmas')[ftfy_range[0]:ftfy_range[1]]
        parent_lemmas = ut.join_sentences(parent, key='lemmas')[parent_range[0]:parent_range[1]]
        ftfy = ut.join_sentences(ut.get_search_string(ftfy, ftfy_index), True)[ftfy_range[0]:ftfy_range[1]]
        parent = ut.join_sentences(parent)[parent_range[0]:parent_range[1]]
        
        ftfy_collocations = wut.segment_collocations(title_lookup, ftfy, ftfy_lemmas, ftfy_pos, collocations_only=True)
        parent_collocations = wut.segment_collocations(title_lookup, parent, parent_lemmas, parent_pos, collocations_only=True)

        #print(ftfy_collocations, parent_collocations)
        
        collocation_pairs = []
        for fc in ftfy_collocations:
            fc_indices = set(title_lookup[fc])
            for pc in parent_collocations:
                pc_indices = set(title_lookup[pc])
                if pc == fc or len(fc_indices & pc_indices):
                    continue
                pairs = wut.get_pairs(link_lookup, fc_indices, pc_indices)
                #print(fc, fc_indices, pc, pc_indices, pairs)
                if len(pairs) > 0:
                    collocation_pairs.append([fc, pc])

        #print(collocation_pairs)
        all_collocations.append(collocation_pairs)

outfile = infile + '.collocations'
if pmi is not None:
    outfile = infile + '.pmi'        
with open(outfile, 'w') as f:
    for collocations in all_collocations:
        print(json.dumps(collocations), file=f)

        
