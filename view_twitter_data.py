from datastorage import DataCollector



indexmax = 9999999

# retrieve the data
twittertweets = DataCollector(filename='twittertweets', overwrite=False)
tweets = twittertweets.retrieve_data(parameter='tweet', indexes=(0, indexmax))
tweet_times = twittertweets.retrieve_data(parameter='tweet_time', indexes=(0, indexmax))





index = -1
print(len(tweets))
print(tweets[index])
print(tweet_times[index])
















