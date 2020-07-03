# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 09:32:48 2020

@author: KIIT
"""

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s
#import time


#consumer key, consumer secret, access token, access secret.
ckey="WEpI5xP3Zh7H8SnxlEUHW3kjS"
csecret="JEAfXnQPyZyE6uQxiG7LbnlStLNIQydN0M2Yb1ug4GNi7gZsxf"
atoken="979879322956701697-NkldT4r2LENLAox4Y02WtOtcmMuQp04"
asecret="6ZpG63QNF9LgILo7sWFRroVllqZh6HXqYAEcb6CEIhxCs"

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)

        tweet = all_data["text"]
        sentiment_value, confidence = s.sentiment(tweet)
        print(tweet, sentiment_value, confidence)
        
        if confidence*100 >= 80:
            output = open("twitter-out.txt", "a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()

        return True

    def on_error(self, status):
        print (status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["car"])