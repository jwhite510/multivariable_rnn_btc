from datastorage import DataCollector
from datetime import datetime
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt




def fix_btc_time(btc_time):

    processed = btc_time.replace('T', ' ')
    if '.' in processed:
        processed = processed.split('.')[0]

    return processed


def convert_time_to_unix(dtstring):

    datetime_object = datetime.strptime(dtstring, '%Y-%m-%d %H:%M:%S')
    time_s = time.mktime(datetime_object.timetuple())

    return time_s


if __name__ == '__main__':


    indexmax = 999999



    # bitcoin data
    bittrexbtcpricedata = DataCollector(filename='bittrexbtcprices', overwrite=False)
    btcprices = bittrexbtcpricedata.retrieve_data(parameter='btcprice', indexes=(0, indexmax))
    btcprice_times = bittrexbtcpricedata.retrieve_data(parameter='btcprice_time', indexes=(0, indexmax))
    btcprice_times = [fix_btc_time(btctime) for btctime in btcprice_times]

    # reddit submissions
    redditsubmissions = DataCollector(filename='redditsubmissions', overwrite=False)
    submissions = redditsubmissions.retrieve_data(parameter='submission', indexes=(0, indexmax))
    submission_times = redditsubmissions.retrieve_data(parameter='submission_time', indexes=(0, indexmax))

    # reddit comments
    redditcomments = DataCollector(filename='redditcomments', overwrite=False)
    comments = redditcomments.retrieve_data(parameter='comment', indexes=(0, indexmax))
    comment_times = redditcomments.retrieve_data(parameter='comment_time', indexes=(0, indexmax))

    # twitter tweets
    twittertweets = DataCollector(filename='twittertweets', overwrite=False)
    tweets = twittertweets.retrieve_data(parameter='tweet', indexes=(0, indexmax))
    tweet_times = twittertweets.retrieve_data(parameter='tweet_time', indexes=(0, indexmax))


    # for timearray in [tweet_times, comment_times, submission_times, btcprice_times]:
    tweet_times = [convert_time_to_unix(element) for element in tweet_times]
    comment_times = [convert_time_to_unix(element) for element in comment_times]
    submission_times = [convert_time_to_unix(element) for element in submission_times]
    btcprice_times = [convert_time_to_unix(element) for element in btcprice_times]


    # convert the bitcoin prices to floats
    btcprices = [float(element) for element in btcprices]

    tweets_wordcount = np.zeros_like(tweets, dtype=float)


    for index, value in enumerate(tweets):


        found_pos = 0
        found_neg = 0

        # positive words
        for searchword in ['buy', 'start', 'million', 'going', 'up', 'increase']:
            if searchword in value.lower():
                found_pos += 1

        # negative words
        for searchword in ['sell', 'decrease', 'drop', 'crash', 'bad']:
            if searchword in value.lower():
                found_neg += 1

        score = found_pos - found_neg

        tweets_wordcount[index] = int(score)




    tweet_times_np = np.array(tweet_times)



    for time1, time2 in zip(btcprice_times[:-1], btcprice_times[1:]):

        indexes = (tweet_times_np > time1) & (tweet_times_np <= time2)
        tweet_times_np[indexes] = time2

    time_sentiment_average = np.zeros_like(btcprice_times[1:], dtype=float)

    for index, time in enumerate(btcprice_times[1:]):

        tweets_from_this_time = tweets_wordcount[tweet_times_np == time]
        """
        NEED TO NORMALIZE FOR TIME HERE!!!!!!!!!!!!!!!!!!!!!
        time steps are not constant
        """
        time_sentiment_average[index] = np.average(tweets_from_this_time)



    fig = plt.figure(constrained_layout=False, figsize=(7, 7))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, :3])
    ax.plot(btcprice_times[1:], time_sentiment_average, color='blue')
    ax.text(0.2, 1, 'Twitter occurance of positive and negative words', transform=ax.transAxes,
            backgroundcolor='white')


    ax = fig.add_subplot(gs[1, :3])
    ax.plot(btcprice_times, btcprices, color='orange')
    ax.text(0.2, 1, 'Bitcoin Price [USD]', transform=ax.transAxes, backgroundcolor='white')


    plt.show()









