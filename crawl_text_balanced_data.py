#%%
import pandas as pd
import requests
import json
import csv
import time
import datetime
import sys
import os
import re

DEBUG=True

#%%
objCount = 0
obj = {}

#%%

def GetPushshiftData(after, before, over_18=False, query="", is_video=False, sub=""):
  url = 'https://api.pushshift.io/reddit/search/submission/?title='+str(query)+'&size=100&after='+str(after)+'&before='+str(before)+'&over_18='+str(over_18)+'&is_video='+str(is_video)+'&subreddit='+str(sub)

  print(url)
  r = requests.get(url)
  data = json.loads(r.text)
  return data['data']

def CollectData(subm):
  objData = list() #list to store data points

  over_18 = 0
  if subm['over_18']:
    over_18 = 1
  post = subm['subreddit'] + '\n' + subm['title'] + '\n' + subm['selftext']
  sub_id = subm['id']
  
  objData.append((sub_id, post over_18))
  obj[sub_id] = objData

def UpdateDataFile(file):
  upload_count = 0
  with open(file, 'w', newline='', encoding='utf-8') as file: 
    a = csv.writer(file, delimiter=',')
    headers = ["POST ID", "post" "NSFW"]
    a.writerow(headers)
    for sub in obj:
      a.writerow(obj[sub][0])
      upload_count+=1
        
    print(str(upload_count) + " submissions have been uploaded")

def DataValid(subm):
  # No media
  if ('media_embed' in subm and subm['media_embed'] != {}) or (
    'secure_media_embed' in subm and subm['secure_media_embed'] != {}):
    return False
  
  # Min number of words in the post  
  if subm['selftext'] == "[deleted]" or subm['selftext'] == "[removed]":
    # print("11")
    return False
  if len(re.sub("[^\w]", " ",  subm['title']).split()) < 5 and subm['selftext'] == "" :   
    # print("22")
    return False
  
  # Should be popular s.t score >= 10 (threshold taken from 
  # https://www.cs.ubc.ca/~nando/540-2013/projects/p38.pdf)
  if subm['score'] < 10 and subm['num_comments'] < 5:
    return False
  
  return True

def GenerateData(nsfw=False):
  # Unix timestamp of date to crawl from 2015/01/01 to 2018/01/01
  after = "1420070400"
  before = "1514764800"

  global objCount, obj

  objCount = 0
  obj = {}

  # Data directory
  dataDir = os.getcwd() + '/data/'
  if not os.path.exists(dataDir):
    os.makedirs(dataDir)

  if nsfw:
    FILENAME = dataDir + 'nsfw_balanced.csv'
  else:
    FILENAME = dataDir + 'sfw_balanced.csv'
  
  data = GetPushshiftData(after, before, over_18=nsfw)

  # Will run until all posts have been gathered 
  # from the 'after' date up until todays date
  while len(data) > 0 and objCount < 500:
    for submission in data:
      if DataValid(submission):
        CollectData(submission)
        objCount+=1
    # Calls getPushshiftData() with the created date of the last submission
    print(len(data))
    print(str(datetime.datetime.fromtimestamp(data[-1]['created_utc'])))
    after = data[-1]['created_utc']
    data = GetPushshiftData(after, before, over_18=nsfw)
      
  print(len(data))

  UpdateDataFile(FILENAME)


#%%
GenerateData(nsfw=True)

#%%
GenerateData(nsfw=False)








#%%
after = "1420070400"
before = "1514764800"
data = GetPushshiftData(after, before, over_18=False)
#%%

print(data[5]['over_18'] == True)


#%%
if DEBUG:
  print(str(len(obj)) + " submissions have added to list")
  print("1st entry is:")
  for i in range(7):
    print(list(obj.values())[6][0][i])


#%%
s = "Maybe Maybe Maybe"
for i in range(100):
  print(len(re.sub("[^\w]", " ",  s).split()))

#%%
for i in range(100):
  print(data[i]['selftext'] == "[removed]")
# %%
print(FILENAME)
updateDataFile(FILENAME)








# %%
for i in range(2):
  print(data[i])

# %%
for i in range(100):
  print(data[i]['selftext']=="")

# %%
xx = 5
def f():
  global xx
  xx += 2
print(xx)
f()
print(xx)
f()
print(xx)

# %%
