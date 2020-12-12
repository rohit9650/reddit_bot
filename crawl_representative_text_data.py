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
from requests.exceptions import ConnectionError

DEBUG=True

#%%
objCount = 0
obj = {}
nsfw_count = 0
total_count = 0

#%%

def GetPushshiftData(after, before, query="", is_video=False, sub=""):
  url = 'https://api.pushshift.io/reddit/search/submission/?title='+str(query)+'&size=1000&after='+str(after)+'&before='+str(before)+'&is_video='+str(is_video)+'&subreddit='+str(sub)
  
  print(url)
  data = None
  r = None
  try:
    r = requests.get(url)
  except ConnectionError as ce:
    print("Response content is not valid JSON")
    return data, False
  
  try:
    data = json.loads(r.text)
  except ValueError:
    print("Response content is not valid JSON")
    return data, False

  return data['data'], True

def CollectData(subm):
  global nsfw_count
  global total_count
  total_count += 1
  
  objData = list() #list to store data points

  over_18 = 0
  if subm['over_18']:
    nsfw_count += 1
    over_18 = 1
  post = ''
  if 'subreddit' in subm:
    post = subm['subreddit']
  post = post + '\n' + subm['title']
  if 'selftext' in subm:
    post = post + '\n' + subm['selftext']
  sub_id = subm['id']
  
  objData.append((sub_id, post, over_18))
  obj[sub_id] = objData

def UpdateDataFile(file):
  upload_count = 0
  with open(file, 'w', newline='', encoding='utf-8') as file: 
    a = csv.writer(file, delimiter=',')
    headers = ["POST ID", "post", "NSFW"]
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
  if 'selftext' in subm:
    if subm['selftext'] == "[deleted]" or subm['selftext'] == "[removed]":
      # print("11")
      return False
    if len(re.sub("[^\w]", " ",  subm['title']).split()) < 5 and subm['selftext'] == "" :   
      # print("22")
      return False
  
  elif len(re.sub("[^\w]", " ",  subm['title']).split()) < 5:
    return False
  
  # Should be popular s.t score >= 10 (threshold taken from 
  # https://www.cs.ubc.ca/~nando/540-2013/projects/p38.pdf)
  if subm['score'] < 10 and subm['num_comments'] < 5:
    return False
  
  return True

def GenerateData():
  # Unix timestamp of date to crawl from 2015/01/01 to 2018/01/01
  after = "1270637661"
  before = "1514764800"

  global objCount, obj

  objCount = 0
  obj = {}

  # Data directory
  dataDir = os.getcwd() + '/data/'
  if not os.path.exists(dataDir):
    os.makedirs(dataDir)

  FILENAME = dataDir + 'text_representative_1.csv'
  
  data, valid = GetPushshiftData(after, before)

  # Will run until all posts have been gathered 
  # from the 'after' date up until todays date
  while len(data) > 0 and objCount < 1000000:
    for submission in data:
      if DataValid(submission):
        CollectData(submission)
        objCount+=1
    # Calls getPushshiftData() with the created date of the last submission
    print(len(data))
    print(str(datetime.datetime.fromtimestamp(data[-1]['created_utc'])))
    after = data[-1]['created_utc']
    data, valid = GetPushshiftData(after, before)

    while not valid:
      print(str(datetime.datetime.fromtimestamp(data[-1]['created_utc'])))
      after = data[-1]['created_utc']
      data, valid = GetPushshiftData(after, before)
      
  print(len(data))

  UpdateDataFile(FILENAME)


#%%
GenerateData()


# %%
print(total_count, nsfw_count)
print('frac: {:.2f}'.format(nsfw_count / total_count))

# %%
dataDir = os.getcwd() + '/data/'
if not os.path.exists(dataDir):
  os.makedirs(dataDir) 

f = dataDir + 'tmp.csv'
with open(f, 'w', newline='', encoding='utf-8') as file: 
  a = csv.writer(file, delimiter=',')
  headers = ["POST ID", "post", "NSFW"]
  a.writerow(headers)
  r = ['a', 'b', 'c']
  a.writerow(r)

with open(f, 'a', newline='', encoding='utf-8') as file: 
  a = csv.writer(file, delimiter=',')
  r = ['b', 'c', 'd']
  a.writerow(r)

# %%
