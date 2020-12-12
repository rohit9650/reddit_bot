#%%
import pandas as pd
# Data directory
dataDir = os.getcwd() + '/data_f/'
if not os.path.exists(dataDir):
  os.makedirs(dataDir)

FILEPATH= dataDir + 'text_test.csv'

df_test = pd.read_csv(FILEPATH)
# df_test = df_test.drop(['POST ID'], axis=1)

FILEPATH= dataDir + 'sfw_balanced.csv'
df_sfw = pd.read_csv(FILEPATH)
df_sfw = df_sfw.loc[df_sfw['NSFW'] == 0]

FILEPATH= dataDir + 'nsfw_balanced.csv'
df_nsfw = pd.read_csv(FILEPATH)
df_nsfw = df_nsfw.loc[df_nsfw['NSFW'] == 1]

#%%
print(df_test.head)
print(df_sfw.head)
print(df_nsfw.head)

# %%
df_balanced = (pd.concat([df_sfw, df_nsfw])).sample(frac=1)
df_balanced.to_csv(dataDir + 'text_balanced.csv', index=False)

#%%
df_rep_sfw = df_sfw.sample(frac=0.85, replace=False, random_state=1)
df_rep_nsfw = df_nsfw.sample(frac=0.1, replace=False, random_state=1)

df_rep = (pd.concat([df_rep_sfw, df_rep_nsfw])).sample(frac=1)
df_rep.to_csv(dataDir + 'text_representative.csv', index=False)
#%%
print(df_rep_sfw)
print(df_res_nsfw)


# %%
df_test.loc[df_test['NSFW'] == 0]

# %%
test = df_balanced[:4]
test.to_csv(dataDir + 'test.csv')

# %%
FILEPATH= dataDir + 'text_representative.csv'
df_= pd.read_csv(FILEPATH)

d1 = df_.loc[df_['NSFW'] == 1]
d2 = df_.loc[df_['NSFW'] == 0]

print(df_)
print(d1)
print(d2)

# %%
df_.loc[df_['POST ID'] == 'POST ID']

# %%
