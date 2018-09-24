from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import twitter_credentials
from datastorage import DataCollector
import json
import datetime






class StdOutListener(StreamListener):

    def __init__(self, datacollectionobject):
        StreamListener.__init__(self)

        self.datacollectionobject = datacollectionobject
        self.tweets = []
        self.tweet_times = []


    def on_data(self, data):

        twitter_json = json.loads(data)

        tweet_text = twitter_json['text']
        tweet_timestamp = twitter_json['timestamp_ms']
        tweet_datetime = datetime.datetime.utcfromtimestamp(int(int(tweet_timestamp))/1000).strftime('%Y-%m-%d %H:%M:%S')

        # print('\ntweet_datetime: \n', tweet_datetime)
        # print('\ntext: \n', tweet_text)

        self.tweets.append(tweet_text)
        self.tweet_times.append(tweet_datetime)

        if len(self.tweets) > 1:
            # append the data
            self.tweets = [element.encode('utf-8') for element in self.tweets]
            self.datacollectionobject.append_data(parameter='tweet', data=self.tweets, times=self.tweet_times)
            self.tweet_times = []
            self.tweets = []
            print('tweet data was appended time: {}'.format(str(datetime.datetime.now())))

        return True

    def on_error(self, status):
        print(status)



if __name__ == "__main__":

    parameters = []
    parameters.append({'name': 'tweet', 'maxlength': 4000})
    twittertweets = DataCollector(filename='twittertweets', parameters=parameters, overwrite=True,
                                   checklength=True)

    # start the twitter instance
    listener = StdOutListener(twittertweets)
    auth = OAuthHandler(twitter_credentials.CONSUMER_KEY,
                        twitter_credentials.CONSUMER_SECRET)

    auth.set_access_token(twitter_credentials.ACCESS_TOKEN,
                          twitter_credentials.ACCESS_TOKEN_SECRET)

    stream = Stream(auth, listener)

    stream.filter(track=['bitcoin'])





















