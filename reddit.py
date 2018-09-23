import praw
import reddit_credentials
import tables
from datastorage import RedditData










if __name__ == "__main__":

    # the data types that this file will contain
    parameters = []
    parameters.append({'name': 'comment', 'maxlength': 30})
    parameters.append({'name': 'submission', 'maxlength': 100})

    redditdata = RedditData(filename='redditdata', parameters=parameters, overwrite=True,
                            checklength=True)


    # collect the comments and times
    comments = ['comment1body', 'comment2body', 'comment3body']
    comment_times = ['comment time 1', 'comment time 2', 'comment time 3']

    redditdata.append_data(parameter='comment', data=comments, times=comment_times)
    redditdata.append_data(parameter='submission', data=comments, times=comment_times)


    data = redditdata.retrieve_data(parameter='comment', indexes=(0, 2))
    print(data)

    data = redditdata.retrieve_data(parameter='comment_time', indexes=(0, 2))
    print(data)

    exit(0)

    #
    # # start the reddit instance
    # reddit = praw.Reddit(client_id=reddit_credentials.CLIENT_ID,
    #                      client_secret=reddit_credentials.CLIENT_SECRET,
    #                      user_agent='agent')
    #
    #
    # subreddit = reddit.subreddit('bitcoin')
    #



