from datastorage import DataCollector



indexmax = 154

# retrieve the data
redditsubmissions = DataCollector(filename='twittertweets', overwrite=False)
submissions = redditsubmissions.retrieve_data(parameter='tweet', indexes=(0, indexmax))
submission_times = redditsubmissions.retrieve_data(parameter='tweet_time', indexes=(0, indexmax))





index = -1

print(submissions[index])
print(submission_times[index])
















