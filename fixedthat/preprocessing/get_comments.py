import datetime
import collections

import reddit_scraper

#given a list of subreddits, get all comments organized by user for a specific day
r = reddit_scraper.reddit

with open('subreddits.txt') as f:
    subreddits = f.read().splitlines()

today = datetime.date.today()
today = datetime.datetime(today.year, today.month, today.day)
yesterday = (today - datetime.timedelta(1) - datetime.datetime.fromtimestamp(0)).total_seconds()
today = (today - datetime.datetime.fromtimestamp(0)).total_seconds()
print(today, yesterday)

author_comments = collections.defaultdict(list)
for sub in subreddits:
    print(sub)
    try:
        s = sorted(r.subreddits.search_by_name(sub, exact=True)[0].submissions(yesterday, today), key=lambda x:x.score, reverse=True)
    except Exception as e:
        print('unable to get', sub, e)
        continue
    
    print(len(s))
    for submission in s[:100]:
        print(submission.author.name, submission.url, submission.selftext)
        author_comments[submission.author.name].append((sub, submission.selftext, submission.title))
        
        try:
            submission.comments.replace_more(limit=0)
            comments = list(submission.comments)
        except Exception as e:
            print('unable to get comments for', sub, e)
            continue
        
        print(len(comments))
        for comment in comments:
            author_comments[comment.author].append((sub, comment.body))
            #print(comment.author, comment.body)

for author in author_comments:
    if len(author_comments[author]) < 1:
        continue
    for comment in author_comments[author]:
        print(author, comment)        
