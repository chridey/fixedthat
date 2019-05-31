'''
script for going through the comments retrieved by get_ftfy.py and matching them with their parent comment
usage: match_ftfy.py <outfile> <infile 1> <infile 2> ...
'''

from __future__ import print_function

#read through all the ftfy, find their parent
#sort by subreddit, then by ups
#write to csv
import sys
import os
import csv
import json
import collections

infiles = sys.argv[2:]
outfile = sys.argv[1]

parents = {}
ftfy = []

for filename in infiles:
    if not filename.endswith('.parents') and not filename.endswith('.ftfy'):
        continue
    print(filename)
    with open(filename) as f:
        for line in f:
            j = json.loads(line)
            if filename.endswith('.parents'):
                if 'selftext' in j:
                    j['body'] = j['selftext']
                    
                line = [j.get('name', ''), j.get('id', ''), j.get('subreddit', ''), j['score'], j['created_utc'], j['body'], j.get('title', '')]
                if 'name' in j:
                    pid = j['name']
                elif 'id' in j:
                    pid = j['id']
                else:
                    continue
                if '_' in pid:
                    pid = pid.split('_')[1]                    

                parents[pid] = line                
                
            elif filename.endswith('.ftfy'):
                ftfy.append([j.get('name', ''), j['subreddit'], j['score'], j['created_utc'], j['body'], j['parent_id'].split('_')[1]])
                
    print(len(ftfy), len(parents))

with open(outfile, 'w') as f:

    writer = csv.writer(f)
    writer.writerow(['name', 'subreddit', 'score', 'created', 'body', 'parent_name', 'parent_id', 'parent_subreddit', 'parent_score', 'parent_created', 'parent_body', 'title'])
    for line in sorted(ftfy, key=lambda x:(x[1], x[2]), reverse=True):
        output = line[:5]

        if line[-1] in parents:
            output += parents[line[-1]]
        elif 't3_'+line[-1] in parents:
            output += parents['t3_'+line[-1]]
        elif 't1_'+line[-1] in parents:
            output += parents['t1_'+line[-1]]
        else:
            continue
        writer.writerow([unicode(s).encode("utf-8") for s in output])            
