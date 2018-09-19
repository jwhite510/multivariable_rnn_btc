import praw
import reddit_credentials



reddit = praw.Reddit(client_id=reddit_credentials.CLIENT_ID,
                     client_secret=reddit_credentials.CLIENT_SECRET,
                     user_agent='agent')


subreddit = reddit.subreddit('bitcoin')
# hot = subreddit.hot(limit=5)
#
# for submission in hot:
#     print(submission.title)


for comment in subreddit.stream.comments():
    try:

        parent_id = str(comment.parent())
        original = reddit.comment(parent_id)

        print('Parent: ')
        print('body: ', original.body)
        print('submission: ', original.submission)
        print(dir(original.submission))

        print('Reply:')
        print(comment.body)


    except Exception as e:
        print(e)



