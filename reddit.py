import praw
import reddit_credentials
import tables
from datastorage import RedditData
import datetime
from multiprocessing import Process



def reddit_collect_submissions(subreddit, datastorage):

    submissions = []
    submission_times = []

    for submission in subreddit.stream.submissions():

        try:

            submission_text = submission.title + '\n\n' + submission.selftext
            submission_timestamp = submission.created_utc
            submission_datetime = datetime.datetime.utcfromtimestamp(submission_timestamp).strftime('%Y-%m-%d %H:%M:%S')

            submissions.append(submission_text)
            submission_times.append(submission_datetime)
            if len(submissions) > 1:
                submissions = [element.encode('utf-8') for element in submissions]
                datastorage.append_data(parameter='submission', data=submissions, times=submission_times)
                print('submission data was appended time: {}'.format(str(datetime.datetime.now())))
                submissions = []
                submission_times = []



        except Exception as e:
            print(e)
            exit(0)
            print('Error occured')





def reddit_collect_comments(subreddit, datastorage):

    comments = []
    comment_times = []

    for comment in subreddit.stream.comments():

        try:

            # comment_text = comment.title + '\n\n' + comment.selftext
            comment_text = comment.body
            comment_timestamp = comment.created_utc
            comment_datetime = datetime.datetime.utcfromtimestamp(comment_timestamp).strftime('%Y-%m-%d %H:%M:%S')

            comments.append(comment_text)
            comment_times.append(comment_datetime)


            # print(comments)

            if len(comments) > 1:
                comments = [element.encode('utf-8') for element in comments]

                datastorage.append_data(parameter='comment', data=comments, times=comment_times)


                print('comments data was appended time: {}'.format(str(datetime.datetime.now())))
                comments = []
                comment_times = []



        except Exception as e:
            print(e)
            exit(0)
            print('Error occured')







if __name__ == "__main__":

    # the data types that this file will contain
    parameters = []
    parameters.append({'name': 'submission', 'maxlength': 10000})
    redditsubmissions = RedditData(filename='redditsubmissions', parameters=parameters, overwrite=True,
                            checklength=True)


    parameters = []
    parameters.append({'name': 'comment', 'maxlength': 4000})
    redditcomments = RedditData(filename='redditcomments', parameters=parameters, overwrite=True,
                                   checklength=True)

    # start the reddit instance
    reddit = praw.Reddit(client_id=reddit_credentials.CLIENT_ID,
                         client_secret=reddit_credentials.CLIENT_SECRET,
                         user_agent='agent')

    subreddit = reddit.subreddit('bitcoin')


    # reddit_collect_submissions(subreddit, redditsubmissions)
    # reddit_collect_comments(subreddit, redditcomments)

    submissionprocess = Process(target=reddit_collect_submissions, args=(subreddit, redditsubmissions,))
    commentprocess = Process(target=reddit_collect_comments, args=(subreddit, redditcomments,))

    submissionprocess.start()
    commentprocess.start()





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




