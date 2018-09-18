import praw
import reddit_credentials



reddit = praw.Reddit(client_id=reddit_credentials.CLIENT_ID,
                     client_secret=reddit_credentials.CLIENT_SECRET,
                     user_agent='agent')


subreddit = reddit.subreddit('bitcoin')
hot = subreddit.hot(limit=5)

for submission in hot:
    print(submission.title)
