from datastorage import DataCollector



indexmax = 154

# retrieve the data
redditsubmissions = DataCollector(filename='redditsubmissions', overwrite=False)
submissions = redditsubmissions.retrieve_data(parameter='submission', indexes=(0, indexmax))
submission_times = redditsubmissions.retrieve_data(parameter='submission_time', indexes=(0, indexmax))


redditcomments = DataCollector(filename='redditcomments', overwrite=False)
comments = redditcomments.retrieve_data(parameter='comment', indexes=(0, indexmax))
comment_times = redditcomments.retrieve_data(parameter='comment_time', indexes=(0, indexmax))



index = -1

# view the comments
print(comments[index])
print(comment_times[index])

# view the submissions
print(submissions[index])
print(submission_times[index])

exit(0)



