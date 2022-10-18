# import numpy as np      ## OFFICIAL CODE, BUT SELF WRITTEN CODE IS USED!!
#
# def build_freqs(tweets, ys):
#     """Build frequencies.
#     Input:
#         tweets: a list of tweets
#         ys: an m x 1 array with the sentiment label of each tweet (either 0 or 1)
#     Output:
#         freqs: a dictionary mapping each (word, sentiment) pair to its frequency
#     """
#     # Convert np array to list since zip needs an iterable.
#     # The squeeze is necessary or the list ends up with one element.
#     # Also note that this is just a NOP if ys is already a list.
#     yslist = np.squeeze(ys).tolist()
#
#     # Start with an empty dictionary and populate it by looping over all tweets
#     # and over all processed words in each tweet.
#     freqs = {}
#     for y, tweet in zip(yslist, tweets):
#         for word in process_tweet(tweet):
#             pair = (word, y)
#             if pair in freqs:
#                 freqs[pair] += 1
#             else:
#                 freqs[pair] = 1
#
#     return freqs
#
#
# import re
# import string
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.tokenize import TweetTokenizer
#
# def process_tweet(tweet):
#     """Process tweet function.
#     Input:
#         tweet: a string containing a tweet
#     Output:
#         tweets_clean: a list of words containing the processed tweet
#
#     """
#     stemmer = PorterStemmer()
#     stopwords_english = stopwords.words('english')
#     # remove stock market tickers like $GE
#     tweet = re.sub(r'\$\w*', '', tweet)
#     # remove old style retweet text "RT"
#     tweet = re.sub(r'^RT[\s]+', '', tweet)
#     # remove hyperlinks
#     tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
#     # remove hashtags
#     # only removing the hash # sign from the word
#     tweet = re.sub(r'#', '', tweet)
#     # tokenize tweets
#     tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
#                                reduce_len=True)
#     tweet_tokens = tokenizer.tokenize(tweet)
#
#     tweets_clean = []
#     for word in tweet_tokens:
#         if (word not in stopwords_english and  # remove stopwords
#                 word not in string.punctuation):  # remove punctuation
#             # tweets_clean.append(word)
#             stem_word = stemmer.stem(word)  # stemming word
#             tweets_clean.append(stem_word)
#
#     return tweets_clean
#
#
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# import nltk
# from nltk.corpus import twitter_samples
# #from utils import build_freqs, process_tweet
#
# import nltk
# #nltk.download('stopwords')
#
# positive_tweets = twitter_samples.strings('positive_tweets.json')
# negative_tweets = twitter_samples.strings('negative_tweets.json')
#
# tweets = positive_tweets + negative_tweets
# labels = np.append(np.ones(len(positive_tweets)) , np.zeros(len(negative_tweets)))
#
# freqs = build_freqs(tweets, labels)
#
# #print(freqs)
#
# text = 'In today’s article, we’re going to look at some negative review response examples and templates. More specifically, we’ll see how negative online reviews impact your business. Then we’ll give you 13 tips for responding to negative reviews with tact.'
#
# keys = process_tweet(text)
#
# #print(keys)
#
#
# data = []
# for word in keys:
#     pos, neg = 0, 0
#
#     if (word, 1) in freqs:
#         pos = freqs[(word , 1)]
#     if (word, 0) in freqs:
#         neg = freqs[(word , 0)]
#
#     data.append([word, pos, neg])
#
# poser, neger = 0 ,0
# for i in data:
#     poser += i[1]
#     neger += i[2]
# print(poser,neger)
# #print('Positive') if (poser > neger) else print('Negative')



#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from utils import process_tweet


# In[3]:


import nltk
from nltk.corpus import twitter_samples


# In[5]:


positive = twitter_samples.strings('positive_tweets.json')
negative = twitter_samples.strings('negative_tweets.json')


# In[6]:


positive


# In[7]:


negative


# In[8]:


full_corpus = positive + negative


# In[9]:


full_corpus


# In[11]:


labels = np.append(np.ones(len(positive)) , np.zeros(len(negative)))


# In[13]:


labels


# In[15]:


print(positive[0])
print(process_tweet(positive[0]))


# In[16]:


freq_table = {}


# In[17]:


for tweet, sentiment in zip(full_corpus, labels):
    token_tweet = process_tweet(tweet)
    for word in token_tweet:
        pair = (word, sentiment)

        if pair not in freq_table:
            freq_table[(word,sentiment)] = 1
        else:
            freq_table[(word,sentiment)] += 1


# In[19]:


print(freq_table)


# In[20]:


dummy_text = 'In today’s article, we’re going to look at some negative review response examples and templates. More specifically, we’ll see how negative online reviews impact your business. Then we’ll give you 13 tips for responding to negative reviews with tact.'


# In[21]:


text = process_tweet(dummy_text)


# In[22]:


poser, neger = 0,0

for word in text:

    if (word, 1) in freq_table:
        poser += freq_table[(word,1)]

    if (word, 0) in freq_table:
        neger += freq_table[(word,0)]
print(poser, neger)


# In[23]:


print("Positive Sentiment!") if poser > neger else print("Negative Sentiment!")

