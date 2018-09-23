import praw
import reddit_credentials
import tables
from datastorage import RedditData
import datetime









if __name__ == "__main__":

    # the data types that this file will contain
    parameters = []
    parameters.append({'name': 'comment', 'maxlength': 30})
    parameters.append({'name': 'submission', 'maxlength': 10000})
    redditdata = RedditData(filename='redditdata', parameters=parameters, overwrite=True,
                            checklength=True)


    # start the reddit instance
    reddit = praw.Reddit(client_id=reddit_credentials.CLIENT_ID,
                         client_secret=reddit_credentials.CLIENT_SECRET,
                         user_agent='agent')

    subreddit = reddit.subreddit('bitcoin')



    submissions = []
    submission_times = []

    for submission in subreddit.stream.submissions():

        try:

            # print('-------------')
            # print('\ntitle:\n', submission.title)
            # print('\nselftext:\n', submission.selftext)
            # print('\nscreated_utc\n', submission.created_utc)
            # timestamp = submission.created_utc
            # print(datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'))
            # print('-------------')

            submission_text = submission.title + '\n\n' + submission.selftext
            submission_timestamp = submission.created_utc
            submission_datetime = datetime.datetime.utcfromtimestamp(submission_timestamp).strftime('%Y-%m-%d %H:%M:%S')


            submissions.append(submission_text)
            submission_times.append(submission_datetime)
            if len(submissions) > 1:
                submissions = [element.encode('utf-8') for element in submissions]
                redditdata.append_data(parameter='submission', data=submissions, times=submission_times)
                print('data was appended time: {}'.format(str(datetime.datetime.now())))
                submissions = []
                submission_times = []
                


        except Exception as e:
            print(e)
            exit(0)
            print('Error occured')




    # # begin collecting data
    #
    # # collect the comments and times
    # comments = ['comment1body', 'comment2body', 'comment3body']
    # comment_times = ['comment time 1', 'comment time 2', 'comment time 3']
    #
    # redditdata.append_data(parameter='comment', data=comments, times=comment_times)
    # redditdata.append_data(parameter='submission', data=comments, times=comment_times)
    #
    #
    # # retrieve the data
    # data = redditdata.retrieve_data(parameter='comment', indexes=(0, 2))
    # print(data)
    #
    # data = redditdata.retrieve_data(parameter='comment_time', indexes=(0, 2))
    # print(data)
    #
    # exit(0)
    #




