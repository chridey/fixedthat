from __future__ import print_function
import bz2
import json
import sys

import requests
import pandas as pd

#http://files.pushshift.io/reddit/comments/RC_2005-12.bz2
#http://files.pushshift.io/reddit/submissions/RS_2006-01.bz2
#bz2.BZ2Decompressor

start_date = sys.argv[1]
end_date = sys.argv[2]

def get_data(date, submissions=False):
    comment_type = 'comments'
    if submissions:
        comment_type = 'submissions'
    url = 'http://files.pushshift.io/reddit/{}/R{}_{}-{:02}.bz2'.format(comment_type, comment_type[0].upper(), date.year, date.month)
    r = requests.get(url)

    print(url)
    try:
        c = bz2.decompress(r.content).splitlines()
    except IOError:
        print('unable to decompress')
        c = []
    print(len(c))
    for line in c:
        try:
            yield json.loads(line)
        except ValueError:
            print('problem with {}'.format(line))
            
#for date in pd.date_range('20120901', '20170901', freq='MS'):
for date in pd.date_range(start_date, end_date, freq='MS'):
    parents = set()
    with open('{}-{}.ftfy'.format(date.year, date.month), 'w') as f:
        for j in get_data(date):
            if 'ftfy' in j['body'].lower():
                parents.add(j['parent_id'])
                print(json.dumps(j), file=f)

    with open('{}-{}.parents'.format(date.year, date.month), 'w') as f:
        for j in get_data(date):
            if 'name' in j and j['name'] in parents or ('id' in j and (j['id'] in parents or 't3_'+j['id'] in parents or 't1_'+j['id'] in parents)):
                print(json.dumps(j), file=f)
        for j in get_data(date, True):
            if 'name' in j and j['name']  in parents or ('id' in j and (j['id'] in parents or 't3_'+j['id'] in parents or 't1_'+j['id'] in parents)):
                print(json.dumps(j), file=f)
                        
