#%%
user_agent=""
client_id=""
client_secret=""
username="-anonymousHunter"
password=""


# %%
import requests
import json
import time
import datetime
import praw
import _pickle as cPickle
from joblib import dump, load
import pandas as pd
import os

def GetPushshiftData(after="", sub=""):
  url = 'https://api.pushshift.io/reddit/search/submission/?&after='+str(after)+'&size=1000&sort_type=created_utc&sort=desc&over_18=False&is_video=False&subreddit='+str(sub)
  
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

def is_url_image(image_url):
  image_formats = ("image/png", "image/jpeg", "image/jpg")
  try:
    r = requests.head(image_url)
  except Exception as e:
    try:
      r = "https://www.reddit.com/" + requests.head(image_url)
    except Exception as e:
      return False
  if 'content-type' in r.headers and r.headers["content-type"] in image_formats:
    return True
  return False

def text_unsafe(text_content):
  # Result directory
  resultMetaDataDir = os.getcwd() + '/results/text/metadate/'
  if not os.path.exists(resultMetaDataDir):
    os.makedirs(resultMetaDataDir)

  representative_count_vect = None
  with open(resultMetaDataDir + 'representative_count_vect.pkl', 'rb') as fid:
    representative_count_vect = cPickle.load(fid)

  # Make text_content df
  text_content = [text_content]


  text_content_representative_count =  representative_count_vect.transform(text_content)

  model_name = "linear_model.LogisticRegression()"
  fname = resultMetaDataDir + model_name + "__Count_vectors__Representative" + ".joblib"
  model = load(fname) 
  result = model.predict(text_content_representative_count)[0]
  if result == 0:
    result = False
  else:
    result = True
  return result

def process_submission(submission):

  # Extraction the textual content of post
  text_content = ''
  if 'subreddit' in submission:
    text_content = text_content + submission['subreddit']
  text_content = text_content + '\n' + submission['title']
  if 'selftext' in submission:
    text_content = text_content + '\n' + submission['selftext']
  # Get the nsfw score
  text_nsfw = text_unsafe(text_content=text_content) 
  
  image_nsfw = False
  image_nsfw_exp = ''
  # Check if post contains an image and get the nsfw score
  if 'url' in submission and is_url_image(submission['url']):
    r = requests.post(
      "https://api.deepai.org/api/nsfw-detector",
      data={
          'image': str(submission['url']),
      },
      headers={'api-key': ''}
    )
    result = r.json()
    if 'err' not in result and result['output']['nsfw_score'] > 0.8:
      image_nsfw = True
      try:
        image_nsfw_exp = result['output']['detections'][0]['name']
      except Exception as e:
        pass
  
  # Check if post is NSFW
  if text_nsfw or image_nsfw:
    # create comment
    comment = "The post seem to be of NSFW category !!"
    if image_nsfw:
      comment = comment + '\n' + image_nsfw_exp
    # post comment
    reddit = praw.Reddit(
      user_agent=user_agent,
      client_id=client_id,
      client_secret=client_secret,
      username=username,
      password=password,
    )
    sub = reddit.submission(id=submission['id'])
    try:
      sub.reply(comment)
      print(text_nsfw, image_nsfw)
      print("Marked NSFW for submission: {}, titled:{}, by author:{} on sub:{}".format(
        submission['id'],
        submission['title'],
        submission['author'],
        submission['subreddit']))
    except:
      pass


#%%
# Start with all the submission from 5 min ago
after = str(int(time.time()) - 500) 
# sub = "testingground4bots"
sub = ""

data, valid = GetPushshiftData(after=after, sub=sub)
while not valid:
   data, valid = GetPushshiftData(after=after, sub=sub)
   # Since data isn't valid let's start more early
   after = str(int(after) - 500)

# %%

while True:
  for submission in data:
    process_submission(submission)
    print('submitted')
  print(str(datetime.datetime.fromtimestamp(data[0]['created_utc'])))
  after = data[0]['created_utc']
  # Halt for some time
  print("Halting for 2 min......")
  time.sleep(120) 
  print("Resume collecting data.......")
  data, valid = GetPushshiftData(after=after, sub=sub)
  while not valid:
    data, valid = GetPushshiftData(after)
    # Since data isn't valid let's halt more
    print("Halting for 2 min......")
    time.sleep(120) 
    print("Resume collecting data.......")



# %%
