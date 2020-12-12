#%%
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn import decomposition, ensemble
import xgboost

import pandas as pd
import os
import csv
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras import layers, models, optimizers
from keras.preprocessing import text, sequence

import _pickle as cPickle
from joblib import dump, load


#%%
# Data directory
dataDir = os.getcwd() + '/data/'
if not os.path.exists(dataDir):
  os.makedirs(dataDir)

# x: input y:output
# Balanced Training data
FILEPATH = dataDir + 'text_balanced.csv'
df_balanced_train = pd.read_csv(FILEPATH)
df_balanced_train = df_balanced_train.drop(['POST ID'], axis=1)
train_balanced_x = df_balanced_train['post']
train_balanced_y = df_balanced_train['NSFW']

# Representative Training data
FILEPATH = dataDir + 'text_representative_new.csv'
df_representative_train = pd.read_csv(FILEPATH)
df_representative_train = df_representative_train.drop(['POST ID'], axis=1)
train_representative_x = df_representative_train['post']
train_representative_y = df_representative_train['NSFW']

# Testing data (Always on representative/real life data)
FILEPATH = dataDir + 'text_test.csv'
df_test = pd.read_csv(FILEPATH)
df_test = df_test.drop(['POST ID'], axis=1)
test_x = df_test['post']
test_y = df_test['NSFW']


#%%
# Result directory
resultDir = os.getcwd() + '/results/text/'
if not os.path.exists(resultDir):
  os.makedirs(resultDir)

# Result (Training data size: 10 million and Test data size: 0.1 million)
FILEPATH = resultDir + 'classic_models_result_testing_final.csv'
result_file =  open(FILEPATH, 'w', newline='', encoding='utf-8')
all_models_result = csv.writer(result_file, delimiter=',')
headers = ["model name", "f1 score", "precision", "recall", "accuracy","confusion  matrix"]
all_models_result.writerow(headers)

# Don't forget at later
# result_file.close()

# Result directory
resultMetaDataDir = os.getcwd() + '/results/text/metadate/'
if not os.path.exists(resultMetaDataDir):
  os.makedirs(resultMetaDataDir)


#%%
# label encode the target variable 
encoder = preprocessing.LabelEncoder()

train_balanced_y = encoder.fit_transform(train_balanced_y)
train_representative_y = encoder.fit_transform(train_representative_y)
test_y = encoder.fit_transform(test_y)


#%%
# create a count vectorizer object 
balanced_count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
balanced_count_vect.fit(train_balanced_x.values)

representative_count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
representative_count_vect.fit(train_representative_x.values)

# save it
with open(resultMetaDataDir + 'balanced_count_vect.pkl', 'wb') as fid:
  cPickle.dump(balanced_count_vect, fid)  
with open(resultMetaDataDir + 'representative_count_vect.pkl', 'wb') as fid:
  cPickle.dump(representative_count_vect, fid)

# transform the training and validation data using count vectorizer object
xtrain_balanced_count =  balanced_count_vect.transform(train_balanced_x)
xtrain_representative_count =  representative_count_vect.transform(train_representative_x)

xtest_balanced_count =  balanced_count_vect.transform(test_x)
xtest_representative_count =  representative_count_vect.transform(test_x)


#%%
########## word level tf-idf

# Balanced
tfidf_balanced_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_balanced_vect.fit(train_balanced_x.values)

xtrain_balanced_tfidf =  tfidf_balanced_vect.transform(train_balanced_x)
xtest_balanced_tfidf =  tfidf_balanced_vect.transform(test_x)

# Representative
tfidf_representative_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_representative_vect.fit(train_representative_x.values)

xtrain_representative_tfidf =  tfidf_representative_vect.transform(train_representative_x)
xtest_representative_tfidf =  tfidf_representative_vect.transform(test_x)

# save it
with open(resultMetaDataDir + 'tfidf_balanced_vect.pkl', 'wb') as fid:
  cPickle.dump(tfidf_balanced_vect, fid)  
with open(resultMetaDataDir + 'tfidf_representative_vect.pkl', 'wb') as fid:
  cPickle.dump(tfidf_representative_vect, fid)


########## ngram level tf-idf 

# Balanced
tfidf_balanced_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_balanced_ngram.fit(train_balanced_x.values)

xtrain_balanced_tfidf_ngram =  tfidf_balanced_ngram.transform(train_balanced_x)
xtest_balanced_tfidf_ngram =  tfidf_balanced_ngram.transform(test_x)

# Representative
tfidf_representative_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_representative_ngram.fit(train_representative_x.values)

xtrain_representative_tfidf_ngram =  tfidf_representative_ngram.transform(train_representative_x)
xtest_representative_tfidf_ngram =  tfidf_representative_ngram.transform(test_x)

# save it
with open(resultMetaDataDir + 'tfidf_balanced_ngram.pkl', 'wb') as fid:
  cPickle.dump(tfidf_balanced_ngram, fid)  
with open(resultMetaDataDir + 'tfidf_representative_ngram.pkl', 'wb') as fid:
  cPickle.dump(tfidf_representative_ngram, fid)


########## characters level tf-idf

# Balanced
tfidf_balanced_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_balanced_ngram_chars.fit(train_balanced_x.values)

xtrain_balanced_tfidf_ngram_chars =  tfidf_balanced_ngram_chars.transform(train_balanced_x)
xtest_balanced_tfidf_ngram_chars =  tfidf_balanced_ngram_chars.transform(test_x)

# Representative
tfidf_representative_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_representative_ngram_chars.fit(train_representative_x.values)

xtrain_representative_tfidf_ngram_chars =  tfidf_representative_ngram_chars.transform(train_representative_x)
xtest_representative_tfidf_ngram_chars =  tfidf_representative_ngram_chars.transform(test_x)

# save it
with open(resultMetaDataDir + 'tfidf_balanced_ngram_chars.pkl', 'wb') as fid:
  cPickle.dump(tfidf_balanced_ngram_chars, fid)  
with open(resultMetaDataDir + 'tfidf_representative_ngram_chars.pkl', 'wb') as fid:
  cPickle.dump(tfidf_representative_ngram_chars, fid)


#%%
def train_model(classifier, feature_vector_train, label, feature_vector_valid, model_name=""):
  # fit the training dataset on the classifier
  classifier.fit(feature_vector_train, label)

  # predict the labels on validation dataset
  predictions = classifier.predict(feature_vector_valid)
  
  # save it
  if metrics.f1_score(predictions, test_y) > 0.6:
    fname = model_name.replace("\n", "")
    fname  = fname.replace(" ", "_")
    fname = resultMetaDataDir + fname + ".joblib"
    dump(classifier, fname) 

  result = [
    model_name,
    round(metrics.f1_score(predictions, test_y), 3),
    round(metrics.precision_score(predictions, test_y), 3),
    round(metrics.recall_score(predictions, test_y), 3),
    round(metrics.accuracy_score(predictions, test_y), 3),
    metrics.confusion_matrix(test_y, predictions)]

  return result


# %%
########## Naive Bayes on Count Vectors

# Balanced
model_name = "naive_bayes.MultinomialNB() \n Count vectors \n Balanced"
result = train_model(naive_bayes.MultinomialNB(), xtrain_balanced_count, train_balanced_y, xtest_balanced_count, model_name=model_name)
all_models_result.writerow(result)

# Representative
model_name = "naive_bayes.MultinomialNB() \n Count vectors \n Representative"
result = train_model(naive_bayes.MultinomialNB(), xtrain_representative_count, train_representative_y, xtest_representative_count, model_name=model_name)
all_models_result.writerow(result)

########## Naive Bayes on Word Level TF IDF Vectors

# Balanced
model_name = "naive_bayes.MultinomialNB() \n WordLevel TF-IDF \n Balanced"
result = train_model(naive_bayes.MultinomialNB(), xtrain_balanced_tfidf, train_balanced_y, xtest_balanced_tfidf, model_name=model_name)
all_models_result.writerow(result)

# Representative
model_name = "naive_bayes.MultinomialNB() \n WordLevel TF-IDF \n Representative"
result = train_model(naive_bayes.MultinomialNB(), xtrain_representative_tfidf, train_representative_y, xtest_representative_tfidf, model_name=model_name)
all_models_result.writerow(result)

########## Naive Bayes on Ngram Level TF IDF Vectors

# Balanced
model_name = "naive_bayes.MultinomialNB() \n N-Gram Vectors \n Balanced"
result = train_model(naive_bayes.MultinomialNB(), xtrain_balanced_tfidf_ngram, train_balanced_y, xtest_balanced_tfidf_ngram, model_name=model_name)
all_models_result.writerow(result)

# Representative
model_name = "naive_bayes.MultinomialNB() \n N-Gram Vectors \n Representative"
result = train_model(naive_bayes.MultinomialNB(), xtrain_representative_tfidf_ngram, train_representative_y, xtest_representative_tfidf_ngram, model_name=model_name)
all_models_result.writerow(result)

########## Naive Bayes on Character Level TF IDF Vectors

# Balanced
model_name = "naive_bayes.MultinomialNB() \n CharLevel Vectors \n Balanced"
result = train_model(naive_bayes.MultinomialNB(), xtrain_balanced_tfidf_ngram_chars, train_balanced_y, xtest_balanced_tfidf_ngram_chars, model_name=model_name)
all_models_result.writerow(result)

# Representative
model_name = "naive_bayes.MultinomialNB() \n CharLevel Vectors \n Representative"
result = train_model(naive_bayes.MultinomialNB(), xtrain_representative_tfidf_ngram_chars, train_representative_y, xtest_representative_tfidf_ngram_chars, model_name=model_name)
all_models_result.writerow(result)


#%%
print("Done NB")





#%%
########## Linear Classifier on Count Vectors

# Balanced
model_name = "linear_model.LogisticRegression() \n Count vectors \n Balanced"
result = train_model(linear_model.LogisticRegression(max_iter=100000), xtrain_balanced_count, train_balanced_y, xtest_balanced_count, model_name=model_name)
all_models_result.writerow(result)

# Representative
model_name = "linear_model.LogisticRegression() \n Count vectors \n Representative"
result = train_model(linear_model.LogisticRegression(max_iter=100000), xtrain_representative_count, train_representative_y, xtest_representative_count, model_name=model_name)
all_models_result.writerow(result)

########## Linear Classifier on Word Level TF IDF Vectors

# Balanced
model_name = "linear_model.LogisticRegression() \n WordLevel TF-IDF \n Balanced"
result = train_model(linear_model.LogisticRegression(max_iter=100000), xtrain_balanced_tfidf, train_balanced_y, xtest_balanced_tfidf, model_name=model_name)
all_models_result.writerow(result)

# Representative
model_name = "linear_model.LogisticRegression() \n WordLevel TF-IDF \n Representative"
result = train_model(linear_model.LogisticRegression(max_iter=100000), xtrain_representative_tfidf, train_representative_y, xtest_representative_tfidf, model_name=model_name)
all_models_result.writerow(result)

########## Linear Classifier on Ngram Level TF IDF Vectors

# Balanced
model_name = "linear_model.LogisticRegression() \n N-Gram Vectors \n Balanced"
result = train_model(linear_model.LogisticRegression(max_iter=100000), xtrain_balanced_tfidf_ngram, train_balanced_y, xtest_balanced_tfidf_ngram, model_name=model_name)
all_models_result.writerow(result)

# Representative
model_name = "linear_model.LogisticRegression() \n N-Gram Vectors \n Representative"
result = train_model(linear_model.LogisticRegression(max_iter=100000), xtrain_representative_tfidf_ngram, train_representative_y, xtest_representative_tfidf_ngram, model_name=model_name)
all_models_result.writerow(result)

########## Linear Classifier on Character Level TF IDF Vectors

# Balanced
model_name = "linear_model.LogisticRegression() \n CharLevel Vectors \n Balanced"
result = train_model(linear_model.LogisticRegression(max_iter=100000), xtrain_balanced_tfidf_ngram_chars, train_balanced_y, xtest_balanced_tfidf_ngram_chars, model_name=model_name)
all_models_result.writerow(result)

# Representative
model_name = "linear_model.LogisticRegression() \n CharLevel Vectors \n Representative"
result = train_model(linear_model.LogisticRegression(max_iter=100000), xtrain_representative_tfidf_ngram_chars, train_representative_y, xtest_representative_tfidf_ngram_chars, model_name=model_name)
all_models_result.writerow(result)


#%%
print("Done Logistic Regression")





# %%
########## RF on Count Vectors

# Balanced
model_name = "ensemble.RandomForestClassifier() \n Count vectors \n Balanced"
result = train_model(ensemble.RandomForestClassifier(), xtrain_balanced_count, train_balanced_y, xtest_balanced_count, model_name=model_name)
all_models_result.writerow(result)

# Representative
model_name = "ensemble.RandomForestClassifier() \n Count vectors \n Representative"
result = train_model(ensemble.RandomForestClassifier(), xtrain_representative_count, train_representative_y, xtest_representative_count, model_name=model_name)
all_models_result.writerow(result)

########## RF on Word Level TF IDF Vectors

# Balanced
model_name = "ensemble.RandomForestClassifier() \n WordLevel TF-IDF \n Balanced"
result = train_model(ensemble.RandomForestClassifier(), xtrain_balanced_tfidf, train_balanced_y, xtest_balanced_tfidf, model_name=model_name)
all_models_result.writerow(result)

# Representative
model_name = "ensemble.RandomForestClassifier() \n WordLevel TF-IDF \n Representative"
result = train_model(ensemble.RandomForestClassifier(), xtrain_representative_tfidf, train_representative_y, xtest_representative_tfidf, model_name=model_name)
all_models_result.writerow(result)

########## RF on Ngram Level TF IDF Vectors

# Balanced
model_name = "ensemble.RandomForestClassifier() \n N-Gram Vectors \n Balanced"
result = train_model(ensemble.RandomForestClassifier(), xtrain_balanced_tfidf_ngram, train_balanced_y, xtest_balanced_tfidf_ngram, model_name=model_name)
all_models_result.writerow(result)

# Representative
model_name = "ensemble.RandomForestClassifier() \n N-Gram Vectors \n Representative"
result = train_model(ensemble.RandomForestClassifier(), xtrain_representative_tfidf_ngram, train_representative_y, xtest_representative_tfidf_ngram, model_name=model_name)
all_models_result.writerow(result)

########## RF on Character Level TF IDF Vectors

# Balanced
model_name = "ensemble.RandomForestClassifier() \n CharLevel Vectors \n Balanced"
result = train_model(ensemble.RandomForestClassifier(), xtrain_balanced_tfidf_ngram_chars, train_balanced_y, xtest_balanced_tfidf_ngram_chars, model_name=model_name)
all_models_result.writerow(result)

# Representative
model_name = "ensemble.RandomForestClassifier() \n CharLevel Vectors \n Representative"
result = train_model(ensemble.RandomForestClassifier(), xtrain_representative_tfidf_ngram_chars, train_representative_y, xtest_representative_tfidf_ngram_chars, model_name=model_name)
all_models_result.writerow(result)


#%%
print("Done RF")






# %%
########## Extereme Gradient Boosting on Count Vectors

# Balanced
model_name = "xgboost.XGBClassifier() \n Count vectors \n Balanced"
result = train_model(xgboost.XGBClassifier(), xtrain_balanced_count, train_balanced_y, xtest_balanced_count, model_name=model_name)
all_models_result.writerow(result)

# Representative
model_name = "xgboost.XGBClassifier() \n Count vectors \n Representative"
result = train_model(xgboost.XGBClassifier(), xtrain_representative_count, train_representative_y, xtest_representative_count, model_name=model_name)
all_models_result.writerow(result)

########## Extereme Gradient Boosting on Word Level TF IDF Vectors

# Balanced
model_name = "xgboost.XGBClassifier() \n WordLevel TF-IDF \n Balanced"
result = train_model(xgboost.XGBClassifier(), xtrain_balanced_tfidf, train_balanced_y, xtest_balanced_tfidf, model_name=model_name)
all_models_result.writerow(result)

# Representative
model_name = "xgboost.XGBClassifier() \n WordLevel TF-IDF \n Representative"
result = train_model(xgboost.XGBClassifier(), xtrain_representative_tfidf, train_representative_y, xtest_representative_tfidf, model_name=model_name)
all_models_result.writerow(result)

########## Extereme Gradient Boosting on Ngram Level TF IDF Vectors

# Balanced
model_name = "xgboost.XGBClassifier() \n N-Gram Vectors \n Balanced"
result = train_model(xgboost.XGBClassifier(), xtrain_balanced_tfidf_ngram, train_balanced_y, xtest_balanced_tfidf_ngram, model_name=model_name)
all_models_result.writerow(result)

# Representative
model_name = "xgboost.XGBClassifier() \n N-Gram Vectors \n Representative"
result = train_model(xgboost.XGBClassifier(), xtrain_representative_tfidf_ngram, train_representative_y, xtest_representative_tfidf_ngram, model_name=model_name)
all_models_result.writerow(result)

########## Extereme Gradient Boosting on Character Level TF IDF Vectors

# Balanced
model_name = "xgboost.XGBClassifier() \n CharLevel Vectors \n Balanced"
result = train_model(xgboost.XGBClassifier(), xtrain_balanced_tfidf_ngram_chars, train_balanced_y, xtest_balanced_tfidf_ngram_chars, model_name=model_name)
all_models_result.writerow(result)

# Representative
model_name = "xgboost.XGBClassifier() \n CharLevel Vectors \n Representative"
result = train_model(xgboost.XGBClassifier(), xtrain_representative_tfidf_ngram_chars, train_representative_y, xtest_representative_tfidf_ngram_chars, model_name=model_name)
all_models_result.writerow(result)


#%%
print("Done XgBoost")
result_file.close()


#%%
## Testing



# %%
