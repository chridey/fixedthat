'''
script for retrieving comments containing FTFY
Usage: get_ftfy.py <start_date> <end_date> <OPTIONAL: keywords>
dates should be formatted YYYYMMDD
default keywords are FTFY
'''

from __future__ import print_function
import bz2
import lzma
import gzip
import json
import sys

import requests
if False:
    import pandas as pd
else:
    import datetime
    
start_date = sys.argv[1]
end_date = sys.argv[2]
if len(sys.argv) > 3:
    keywords = sys.argv[3].split(',')
else:
    keywords = ['ftfy']
    
def get_data(date, submissions=False):
    comment_type = 'comments'
    if submissions:
        comment_type = 'submissions'
    url = 'http://files.pushshift.io/reddit/{}/R{}_{}-{:02}.bz2'.format(comment_type, comment_type[0].upper(), date.year, date.month)
    r = requests.get(url)

    print(url)
    try:
        c = bz2.decompress(r.content).decode('utf-8').splitlines()
    except IOError:
        if submissions:
            url = 'http://files.pushshift.io/reddit/{}/R{}_v2_{}-{:02}.xz'.format(comment_type, comment_type[0].upper(), date.year, date.month)
            r = requests.get(url)

            print(url)
        try:
            c = lzma.decompress(r.content).decode('utf-8').splitlines()
        except IOError:
            print('unable to decompress')
            c = []
    print(len(c))
    for line in c:
        try:
            yield json.loads(line)
        except ValueError:
            print('problem with {}'.format(line))

start_date = datetime.datetime.strptime(start_date, "%Y%m%d")
end_date = datetime.datetime.strptime(end_date, "%Y%m%d")
date = start_date
while date <= end_date:
    parents = set()
    count = 0
    with gzip.open('{}-{}.{}.gz'.format(date.year, date.month, '.'.join(keywords)), 'w') as f:
        for j in get_data(date):
            body = j['body'].lower()
            for k in keywords:
                index = body.find(k)
                if index != -1 and (index+len(k) >= len(body) or not body[index+len(k)].isalnum()) and (index == 0 or not body[index-1].isalnum()):
                    count += 1
                    parents.add(j['parent_id'])
                    f.write(bytes(json.dumps(j) + "\n", 'utf-8')) #print(json.dumps(j), file=f)
                    break

    print(count)
    
    with gzip.open('{}-{}.{}.parents.gz'.format(date.year, date.month, '.'.join(keywords)), 'w') as f:
        for j in get_data(date):
            try:
                if 'name' in j and j['name'] in parents or ('id' in j and (j['id'] in parents or 't3_'+j['id'] in parents or 't1_'+j['id'] in parents)):
                    f.write(bytes(json.dumps(j) + "\n", 'utf-8')) #print(json.dumps(j), file=f)
            except Exception:
                continue
        for j in get_data(date, True):
            try:
                if 'name' in j and j['name']  in parents or ('id' in j and (j['id'] in parents or 't3_'+j['id'] in parents or 't1_'+j['id'] in parents)):
                    f.write(bytes(json.dumps(j) + "\n", 'utf-8')) #print(json.dumps(j), file=f)
            except Exception:
                continue
            
    year = date.year
    month = date.month + 1
    if month > 12:
        month = 1
        year += 1
    date = datetime.datetime(year, month, 1)
    
