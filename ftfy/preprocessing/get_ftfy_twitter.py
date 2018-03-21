from __future__ import print_function

import sys
import datetime
import time

import tweepy

consumer_key = 'UM2EmGnwaBNPzHBJy9lx8Cfti'
consumer_secret = 'LxlFrj7AWWoAqSKfaBKVh7gM9eq9wQMb1kg1g5RhUe8qXY1Cyn'
access_token = '3344526328-Dqd2iaOE5DWYQbuMl8xdAtM5jZVWwuxVeVt7aNg'
access_token_secret = 'a8WRbcfk1mhwYqQgFzPTEv7HedVLYrrVypGmKtB0BgoLL'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#for status in tweepy.Cursor(api.search, q='ftfy', tweet_mode='extended').items():
#    print(status._json)
#public_tweets = api.home_timeline()
#for tweet in public_tweets:
#    print tweet.text

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    td = datetime.datetime.today()
    filename = '{}.twitter.ftfy.jsonlist'.format(td.strftime("%m%d%y"))

search = 'ftfy'       
c = tweepy.Cursor(api.search,
                  q=search,
                  tweet_mode='extended',
                  include_entities=True).items()

while True:
    try:
        tweet = c.next()
        print(tweet._json)
    except tweepy.TweepError:
        time.sleep(60 * 15)
        continue
    except StopIteration:
        break
    
    '''
    with open(filename, 'w') as f:
        try:
            tweet = c.next()
            print(tweet._json, file=f)
        except tweepy.TweepError:
            time.sleep(60 * 15)
            continue
        except StopIteration:
            break
    '''
