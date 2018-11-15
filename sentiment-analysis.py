#Author: Adish Jain
#A Sentiment Analysis of Donald Trump's Tweets 
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile

# Ensure that Pandas shows at least 280 characters in columns, so we can see full tweets
pd.set_option('max_colwidth', 280)

%matplotlib inline
plt.style.use('fivethirtyeight')
import seaborn as sns
sns.set()
sns.set_context("talk")
import re
import tweepy
from tweepy import TweepError
import json
import logging


key_file = 'keys.json'
with open(key_file) as f:
    keys = json.load(f)
try:
    auth = tweepy.OAuthHandler(keys["consumer_key"], keys["consumer_secret"])
    auth.set_access_token(keys["access_token"], keys["access_token_secret"])
    api = tweepy.API(auth)
    print("Your username is:", api.auth.get_username())
except TweepError as e:
    logging.warning("There was a Tweepy error. Double check your API keys and try again.")
    logging.warning(e)


#===================FUNCTIONS FOR TWITTER API==========================#
def load_keys(path):
    """Loads your Twitter authentication keys from a file on disk.

    Args:
        path (str): The path to your key file.  The file should
          be in JSON format and look like this (but filled in):
            {
                "consumer_key": "<your Consumer Key here>",
                "consumer_secret":  "<your Consumer Secret here>",
                "access_token": "<your Access Token here>",
                "access_token_secret": "<your Access Token Secret here>"
            }
    Returns:
        dict: A dictionary mapping key names (like "consumer_key") to
          key values."""

    with open(path, "r") as f:
        keys = json.load(f)
    return keys

def download_recent_tweets_by_user(user_account_name, keys):
    """Downloads tweets by one Twitter user.

    Args:
        user_account_name (str): The name of the Twitter account
          whose tweets will be downloaded.
        keys (dict): A Python dictionary with Twitter authentication
          keys (strings), like this (but filled in):
            {
                "consumer_key": "<your Consumer Key here>",
                "consumer_secret":  "<your Consumer Secret here>",
                "access_token": "<your Access Token here>",
                "access_token_secret": "<your Access Token Secret here>"
            }

    Returns:
        list: A list of Dictonary objects, each representing one tweet."""

    try:
        auth = tweepy.OAuthHandler(keys["consumer_key"], keys["consumer_secret"])
        auth.set_access_token(keys["access_token"], keys["access_token_secret"])
        api = tweepy.API(auth)
    except TweepError as e:
        logging.warning("There was a Tweepy error. Double check your API keys and try again.")
        logging.warning(e)
    user_tweets = [t._json for t in tweepy.Cursor(api.user_timeline, id=user_account_name,
                                             tweet_mode='extended').items()]
    return user_tweets


def save_tweets(tweets, path):
    """Saves a list of tweets to a file in the local filesystem.

    This function makes no guarantee about the format of the saved
    tweets, **except** that calling load_tweets(path) after
    save_tweets(tweets, path) will produce the same list of tweets
    and that only the file at the given path is used to store the
    tweets.  (That means you can implement this function however
    you want, as long as saving and loading works!)

    Args:
        tweets (list): A list of tweet objects (of type Dictionary) to
          be saved.
        path (str): The place where the tweets will be saved.

    Returns:
        None"""
    with open(path, "w") as f:
        json.dump(tweets, f)

def load_tweets(path):
    """Loads tweets that have previously been saved.

    Calling load_tweets(path) after save_tweets(tweets, path)
    will produce the same list of tweets.

    Args:
        path (str): The place where the tweets were be saved.

    Returns:
        list: A list of Dictionary objects, each representing one tweet."""

    with open(path, "r") as f:
        tweets = list(json.load(f))
    return tweets

def get_tweets_with_cache(user_account_name, keys_path):
    """Get recent tweets from one user, loading from a disk cache if available.

    The first time you call this function, it will download tweets by
    a user.  Subsequent calls will not re-download the tweets; instead
    they'll load the tweets from a save file in your local filesystem.

    Args:
        user_account_name (str): The Twitter handle of a user, without the @.
        keys_path (str): The path to a JSON keys file in your filesystem.
    """

    tweets_save_path = user_account_name + "_tweets.json"
    tweets = download_recent_tweets_by_user(user_account_name, keys)
    if not Path(tweets_save_path).is_file():
        save_tweets(tweets, tweets_save_path)
    load_tweets(tweets_save_path)
    return tweets


#=======================DATA AGGREGATION=========================#
trump_tweets = get_tweets_with_cache("realdonaldtrump", key_file)
# Download the dataset
from utils import fetch_and_cache
data_url = 'http://www.ds100.org/fa18/assets/datasets/old_trump_tweets.json.zip'
file_name = 'old_trump_tweets.json.zip'

dest_path = fetch_and_cache(data_url=data_url, file=file_name)
print(f'Located at {dest_path}')
my_zip = zipfile.ZipFile(dest_path, 'r')
with my_zip.open("old_trump_tweets.json", "r") as f:
    old_trump_tweets = json.load(f)

ids_in_trump_tweets = [x["id"] for x in trump_tweets]
all_tweets = trump_tweets + [dic for dic in old_trump_tweets if dic["id"] not in ids_in_trump_tweets]

#======================DATA CLEANING==============================#
def extract_text(dictionary):
    if "text" in dictionary.keys():
        return dictionary["text"]
    else:
        return dictionary["full_text"]

trump = pd.DataFrame()

ids = [x["id"] for x in all_tweets]
times = [pd.to_datetime(x["created_at"]) for x in all_tweets]
sources = [x["source"] for x in all_tweets]
texts = [extract_text(x) for x in all_tweets]
retweet_counts = [x["retweet_count"] for x in all_tweets]

trump["id"] = ids
trump["time"] = times
trump["source"] = sources
trump["text"] = texts
trump["retweet_count"] = retweet_counts
trump = trump.set_index("id")
trump.sort_index(inplace=True)
trump.head()

#=====================TWEET SOURCE ANALYSIS=======================#
trump['source'].unique()
trump['source'] = trump['source'].str.replace('<[^>]*>', '')
from datetime import datetime
ELEC_DATE = datetime(2016, 11, 8)
INAUG_DATE = datetime(2017, 1, 20)
assert set(trump[(trump['time'] > ELEC_DATE) & (trump['time'] < INAUG_DATE) ]['source'].unique()) == set(['Twitter Ads',
 'Twitter Web Client',
 'Twitter for Android',
 'Twitter for iPhone'])
trump['source'].value_counts().plot(kind="bar")
plt.ylabel("Number of Tweets")

trump['est_time'] = (
    trump['time'].dt.tz_localize("UTC") # Set initial timezone to UTC
                 .dt.tz_convert("EST") # Convert to Eastern Time
)
trump['hour'] = trump["est_time"].dt.hour + trump["est_time"].dt.minute / 60 + trump["est_time"].dt.second / 3600

#PLOTS TO UNDERSTAND WHAT HOURS OF THE DAY TRUMP USES HIS ANDROID/IPHONE DEVICES
android_iphone_data = trump[(trump["source"] == "Twitter for Android") | (trump["source"] == "Twitter for iPhone")]
sns.distplot(android_iphone_data[android_iphone_data["source"] == "Twitter for iPhone"]["hour"], hist=False, label="iPhone")
sns.distplot(android_iphone_data[android_iphone_data["source"] == "Twitter for Android"]["hour"], hist=False, label="Android")
plt.xlabel("hour")
plt.ylabel("fraction")
plt.legend();

android_iphone_data["year"] = trump["est_time"].dt.year
android_iphone_data_2016 = android_iphone_data[android_iphone_data["year"] == 2016]
sns.distplot(android_iphone_data_2016[android_iphone_data["source"] == "Twitter for iPhone"]["hour"], hist=False, label="iPhone")
sns.distplot(android_iphone_data_2016[android_iphone_data["source"] == "Twitter for Android"]["hour"], hist=False, label="Android")
plt.xlabel("hour")
plt.ylabel("fraction")
plt.legend();

#PLOTS TO UNDERSTAND DEVICE USAGE ACROSS YEARS
import datetime
def year_fraction(date):
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length


trump['year'] = trump['time'].apply(year_fraction)
android_iphone_data_revised = trump[(trump["source"] == "Twitter for Android") | (trump["source"] == "Twitter for iPhone")]
sns.distplot(android_iphone_data_revised[android_iphone_data_revised["source"] == "Twitter for iPhone"]["year"], label="iPhone")
sns.distplot(android_iphone_data_revised[android_iphone_data_revised["source"] == "Twitter for Android"]["year"], label="Android")
plt.xlabel("year")
plt.legend();

#=====================VADER SENTIMENT ANALYSIS=======================#
sent = pd.read_csv("vader_lexicon.txt", delimiter="\t", header=None)
sent = sent.drop([3, 2], axis=1)
sent.columns = ['token', 'polarity']
sent.set_index("token", inplace=True)

trump["text"] = trump["text"].str.lower()

punct_re = r'[^\w\s]'
trump['no_punc'] = trump["text"].str.replace(punct_re, " ")

trump["no_punc"] = trump["no_punc"].str.replace("\s+", " ").str.strip()
tidy_format = trump["no_punc"].str.split(" ", expand=True)
tidy_format = tidy_format.stack()
tidy_format = tidy_format.reset_index(level=1)
tidy_format.columns = ["num", "word"]

trump_and_lexicon = tidy_format.merge(right=sent, how="left", left_on="word", right_index=True)
trump_and_lexicon = trump_and_lexicon.drop(["num", "word"], axis=1)
trump_and_lexicon = trump_and_lexicon.groupby("id").sum()
trump_and_lexicon = trump_and_lexicon.sort_index()
trump['polarity'] = trump_and_lexicon["polarity"]

#SEEING HOW REFERENCES TO DIFFERENT NEWS OUTLETS CORRELATES WITH TWEET SENTIMENT
trump_nyt = trump[trump["no_punc"].str.contains("nyt")]
trump_fox = trump[trump["no_punc"].str.contains("fox")]

sns.distplot(trump_nyt["polarity"], label="nyt")
sns.distplot(trump_fox["polarity"], label="fox")

plt.ylabel("proportion")
plt.legend();

#=====================ENGAGEMENT ANALYSIS==============================#
words_and_trump = tidy_format.merge(trump, how="inner", left_index=True, right_index=True)
sorted_words_and_trump = words_and_trump.sort_values("retweet_count", ascending=False)
counts = sorted_words_and_trump.groupby("word").count()
filter_by = counts[counts["num"] >= 25].index.values
filtered_sorted_words_and_trump = sorted_words_and_trump.loc[sorted_words_and_trump['word'].isin(filter_by)]
final_df = filtered_sorted_words_and_trump.groupby("word").mean().sort_values("retweet_count", ascending=False)
top_20 = final_df.drop(["num", "hour", "year", "polarity"], axis=1).iloc[0:20]
#VISUALIZING WHICH WORDS TRUMP USES IN TWEETS THAT GET MOST ENGAGEMENT
top_20['retweet_count'].sort_values().plot.barh(figsize=(10, 8));

#REGRESSING RETWEET COUNT AS A FUNCTION OF POLARITY
sns.regplot(x="polarity", y="retweet_count", data=final_df, marker="+");
