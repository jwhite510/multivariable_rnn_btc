from datastorage import RedditData




redditdata = RedditData(filename='redditdata', overwrite=False)


# retrieve the data
submissions = redditdata.retrieve_data(parameter='submission', indexes=(0, 154))


submission_times = redditdata.retrieve_data(parameter='submission_time', indexes=(0, 154))

index = 1
print(len(submissions))
exit(0)
print(submissions[index])
print(submission_times[index])





